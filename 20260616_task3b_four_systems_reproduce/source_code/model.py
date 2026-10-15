import csv
import os
import json
import glob
import re
import math
import hashlib
import pickle
import sqlite3
from collections import defaultdict, OrderedDict

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import box_iou, MultiScaleRoIAlign
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import griddata

# Assumes acoustic_features is accessible in your path
from acoustic_features import wav_path_from_seq_dir
from fusion_blocks import (
    AcousticFeaturePyramid,
    CrossModalFPNFusion,
    ResNetConformerFPNAdapter,
    RoICMAFFusionHead,
)


# ── 1. SEQUENCE / FRAME UTILITIES ────────────────────────────────────────────

def json_to_seq_name(json_path: str) -> str:
    stem = os.path.splitext(os.path.basename(json_path))[0]
    return re.sub(r"_std$", "", stem, flags=re.IGNORECASE)

def candidate_seq_dirs(frames_base: str, split_dir: str, seq_name: str) -> list:
    return [
        os.path.join(frames_base, split_dir, seq_name),
        os.path.join(frames_base, split_dir, "3", f"{seq_name}.mp4"),
        os.path.join(frames_base, split_dir, "4", f"{seq_name}.mp4"),
        os.path.join(frames_base, split_dir, "6", f"{seq_name}.mp4"),
    ]


def existing_seq_dir(frames_base: str, split_dir: str, seq_name: str) -> str | None:
    for path in candidate_seq_dirs(frames_base, split_dir, seq_name):
        if os.path.isdir(path):
            return path
    return None

def frame_path(seq_dir: str, seq_name: str, idx: int) -> str:
    candidates = [
        os.path.join(seq_dir, f"{seq_name}_{idx:04d}.png"),
        os.path.join(seq_dir, f"{idx:04d}.png"),
        os.path.join(seq_dir, f"{idx:05d}.png"),
        os.path.join(seq_dir, f"{idx:04d}.jpg"),
        os.path.join(seq_dir, f"{idx:05d}.jpg"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]

def scan_available_frames(seq_dir: str, seq_name: str) -> list:
    patterns = [
        os.path.join(seq_dir, f"{seq_name}_*.png"),
        os.path.join(seq_dir, f"{seq_name}_*.jpg"),
        os.path.join(seq_dir, "*.png"),
        os.path.join(seq_dir, "*.jpg"),
    ]
    out = []
    for pattern in patterns:
        for p in sorted(glob.glob(pattern)):
            name = os.path.basename(p)
            m = re.search(r"_(\d{4,5})\.(?:png|jpg)$", name, flags=re.IGNORECASE)
            if m:
                out.append(int(m.group(1)))
                continue
            stem = os.path.splitext(name)[0]
            if stem.isdigit():
                out.append(int(stem))
    return sorted(set(out))


def _segmentation_to_triplets(segmentation) -> np.ndarray:
    parts = []
    for sub in segmentation:
        if not sub:
            continue
        arr = np.asarray(sub, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 3:
            continue
        parts.append(arr[:, :3])
    if not parts:
        return np.empty((0, 3), dtype=np.float32)
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts, axis=0)


def _render_dense_poly(
    triplets,
    src_w: int,
    src_h: int,
    canvas_w: int = 100,
    canvas_h: int = 50,
) -> np.ndarray:
    pts = np.asarray(triplets, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3:
        return np.zeros((canvas_h, canvas_w), dtype=np.float32)

    canvas_ext_w = canvas_w * 2
    canvas_ext = np.zeros((canvas_h, canvas_ext_w), dtype=np.float32)
    sx = canvas_w / float(src_w)
    sy = canvas_h / float(src_h)
    xs = pts[:, 0] * sx
    ys = pts[:, 1] * sy
    vals = pts[:, 2]

    sorted_xs = np.sort(xs)
    gaps = np.diff(sorted_xs)
    if len(gaps) > 0:
        max_gap_idx = int(np.argmax(gaps))
        if gaps[max_gap_idx] > canvas_w / 2:
            split_threshold = sorted_xs[max_gap_idx]
            xs = np.where(xs <= split_threshold, xs + canvas_w, xs)

    degenerate = len(pts) < 3
    if not degenerate:
        try:
            grid_x, grid_y = np.meshgrid(
                np.arange(canvas_ext_w), np.arange(canvas_h)
            )
            dense = griddata(
                (xs, ys),
                vals,
                (grid_x, grid_y),
                method="linear",
                fill_value=0.0,
            )
            canvas_ext = np.nan_to_num(dense, nan=0.0)
            canvas_ext = np.clip(canvas_ext, 0.0, None)
        except Exception:
            degenerate = True

    if degenerate:
        for x, y, val in zip(xs, ys, vals):
            xi = int(np.clip(round(float(x)), 0, canvas_ext_w - 1))
            yi = int(np.clip(round(float(y)), 0, canvas_h - 1))
            canvas_ext[yi, xi] = max(canvas_ext[yi, xi], float(val))

    return np.maximum(canvas_ext[:, :canvas_w], canvas_ext[:, canvas_w:])


def _render_segmentation_dense(
    segmentation,
    img_w: int,
    img_h: int,
    canvas_w: int = 100,
    canvas_h: int = 50,
) -> np.ndarray:
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    for poly in segmentation:
        if not poly:
            continue
        canvas = np.maximum(
            canvas,
            _render_dense_poly(poly, img_w, img_h, canvas_w=canvas_w, canvas_h=canvas_h),
        )
    peak = float(canvas.max())
    if peak > 1e-8:
        canvas = canvas / peak
    if (canvas_w, canvas_h) == (img_w, img_h):
        return canvas.astype(np.float32, copy=False)
    pil = Image.fromarray(canvas.astype(np.float32), mode="F")
    return np.asarray(
        pil.resize((img_w, img_h), Image.BILINEAR),
        dtype=np.float32,
    ).clip(0.0, 1.0)


def _wrap_azimuth(deg: float) -> float:
    return ((float(deg) + 180.0) % 360.0) - 180.0


def _cart_to_json_xy(x: float, y: float, z: float, img_w: int, img_h: int) -> tuple[float, float]:
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1e-8:
        return img_w * 0.5, img_h * 0.5
    x, y, z = x / norm, y / norm, z / norm
    az = math.degrees(math.atan2(y, x))
    el = math.degrees(math.atan2(z, math.sqrt(x * x + y * y)))
    # STAIRS26 JSON convention mirrors SELD azimuth: az=0 -> x=180,
    # positive azimuth moves left, not right.
    px = (float(img_w) * 0.5 - _wrap_azimuth(az)) % float(img_w)
    py = float(img_h) * 0.5 - el
    return px, float(np.clip(py, 0.0, float(img_h - 1)))

def get_sequence_infos(split_keyword: str, labels_base: str, frames_base: str) -> list:
    infos = []
    if not os.path.isdir(labels_base) or not os.path.isdir(frames_base):
        return infos

    allow_frameless = os.environ.get("DCASE2026_ALLOW_FRAMELESS", "").strip().lower() in {
        "1", "true", "yes", "on"
    }
    skipped = defaultdict(list)
    split_key = str(split_keyword or "").upper().replace("-", "_")
    include_re = (
        os.environ.get(f"DCASE2026_{split_key}_SPLIT_REGEX", "")
        or os.environ.get("DCASE2026_SPLIT_INCLUDE_REGEX", "")
    ).strip()
    exclude_re = (
        os.environ.get(f"DCASE2026_{split_key}_SPLIT_EXCLUDE_REGEX", "")
        or os.environ.get("DCASE2026_SPLIT_EXCLUDE_REGEX", "")
    ).strip()

    for split_dir in sorted(os.listdir(labels_base)):
        if split_keyword not in split_dir:
            continue
        if include_re and re.search(include_re, split_dir) is None:
            continue
        if exclude_re and re.search(exclude_re, split_dir) is not None:
            continue
        split_path = os.path.join(labels_base, split_dir)
        if not os.path.isdir(split_path):
            continue

        for json_file in sorted(glob.glob(os.path.join(split_path, "*.json"))):
            seq_name = json_to_seq_name(json_file)
            seq_dir = existing_seq_dir(frames_base, split_dir, seq_name)
            if seq_dir is None:
                if allow_frameless:
                    seq_dir = candidate_seq_dirs(frames_base, split_dir, seq_name)[0]
                else:
                    skipped[split_dir].append(seq_name)
                    continue
            infos.append((json_file, seq_dir, seq_name))

    if skipped:
        for split_dir, seq_names in sorted(skipped.items()):
            preview = ", ".join(seq_names[:12])
            suffix = "" if len(seq_names) <= 12 else ", ..."
            print(
                f"[Frames] Skipping {len(seq_names)} frameless sequences in {split_dir}: "
                f"{preview}{suffix}"
            )
    return infos


# ── 2. DATASET & CACHING ──────────────────────────────────────────────────────

_DB_SCHEMA_VERSION = "v4_stream_sqlite"

def _dataset_cache_fingerprint(sequence_infos: list) -> str:
    h = hashlib.md5()
    h.update(_DB_SCHEMA_VERSION.encode())   
    for json_path, seq_dir, seq_name in sorted(sequence_infos):
        if os.path.exists(json_path):
            mtime = str(os.path.getmtime(json_path))
        else:
            mtime = "missing"
        h.update(f"{json_path}:{mtime}:{seq_dir}:{seq_name}".encode())
    return h.hexdigest()[:16]


def _iter_json_array_items(json_path: str, key: str, chunk_size: int = 1 << 20):
    """Yield objects from a top-level JSON array field without fp.read()."""
    decoder = json.JSONDecoder()
    marker = f'"{key}"'

    with open(json_path, "r", encoding="utf-8") as f:
        buf = ""
        state = "key"

        while state != "items":
            chunk = f.read(chunk_size)
            if not chunk:
                raise ValueError(f"Could not find JSON array key {key!r} in {json_path}")
            buf += chunk

            progressed = True
            while progressed:
                progressed = False
                if state == "key":
                    pos = buf.find(marker)
                    if pos >= 0:
                        buf = buf[pos + len(marker):]
                        state = "colon"
                        progressed = True
                    elif len(buf) > len(marker) + 128:
                        buf = buf[-(len(marker) + 128):]
                elif state == "colon":
                    pos = buf.find(":")
                    if pos >= 0:
                        buf = buf[pos + 1:]
                        state = "array"
                        progressed = True
                elif state == "array":
                    pos = buf.find("[")
                    if pos >= 0:
                        buf = buf[pos + 1:]
                        state = "items"
                        progressed = True

        while True:
            pos = 0
            while True:
                if pos >= len(buf):
                    chunk = f.read(chunk_size)
                    if not chunk:
                        return
                    buf += chunk
                    continue
                ch = buf[pos]
                if ch in " \t\r\n,":
                    pos += 1
                    continue
                if ch == "]":
                    return
                break

            if pos > 0:
                buf = buf[pos:]

            while True:
                try:
                    item, end = decoder.raw_decode(buf)
                except json.JSONDecodeError:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        raise
                    buf += chunk
                    continue
                yield item
                buf = buf[end:]
                break


def _iter_json_annotations(json_path: str):
    try:
        import ijson  # type: ignore
    except ImportError:
        yield from _iter_json_array_items(json_path, "annotations")
        return

    with open(json_path, "rb") as f:
        yield from ijson.items(f, "annotations.item")


class EnergySegDataset(Dataset):
    DEFAULT_CACHE_DIR = ".dataset_cache"

    def __init__(
        self,
        sequence_infos: list,
        frames_base: str,
        mic_base: str,
        acoustic_extractor,
        frames_per_epoch: int = None,
        img_w: int = 360,
        img_h: int = 180,
        dist_norm: float = 500.0,
        cache_max_size: int = 200,
        augmentor=None,
        gpu_rgb_pipeline: bool = False,
        fixed_subset: bool = False,
        subset_seed: int = 0,
        return_meta: bool = False,
        annotation_target_mode: str = "sparse",
        teacher23_csv_dir: str | None = None,
        teacher23_sigma_px: float = 4.0,
        teacher23_fallback_gt_peak: bool = False,
        class_balanced_sampling: bool | None = None,
        class_balanced_fraction: float = 0.5,
        class_balanced_classes: set[int] | list[int] | tuple[int, ...] | None = None,
    ):
        self.frames_base        = frames_base
        self.mic_base           = mic_base
        self.acoustic_extractor = acoustic_extractor
        self.frames_per_epoch   = frames_per_epoch
        self.img_w              = img_w
        self.img_h              = img_h
        self.dist_norm          = dist_norm
        self.cache_max_size     = cache_max_size
        self.augmentor          = augmentor
        self.gpu_rgb_pipeline   = gpu_rgb_pipeline
        self.fixed_subset       = bool(fixed_subset)
        self.subset_seed        = int(subset_seed)
        self.return_meta        = bool(return_meta)
        self.teacher23_csv_dir  = os.path.abspath(os.path.expanduser(teacher23_csv_dir)) if teacher23_csv_dir else ""
        self.teacher23_sigma_px = float(max(teacher23_sigma_px, 1e-3))
        self.teacher23_fallback_gt_peak = bool(teacher23_fallback_gt_peak)
        self.class_balanced_sampling = (
            self.augmentor is not None
            if class_balanced_sampling is None
            else bool(class_balanced_sampling)
        )
        self.class_balanced_fraction = float(np.clip(class_balanced_fraction, 0.0, 1.0))
        self.class_balanced_classes = (
            {int(c) for c in class_balanced_classes}
            if class_balanced_classes is not None
            else None
        )
        self.teacher23_by_seq_frame = self._load_teacher23_csv_dir(self.teacher23_csv_dir)
        self.annotation_target_mode = str(annotation_target_mode or "sparse").strip().lower()
        if self.annotation_target_mode not in {"sparse", "dense_eval"}:
            raise ValueError(
                f"Unsupported annotation_target_mode={annotation_target_mode!r}; "
                "expected 'sparse' or 'dense_eval'."
            )
        self._fixed_current_indices = None

        self.frame_cache    = OrderedDict()

        self.seq_map        = {}
        self.all_samples    = []
        self.current_indices= []
        self.class_to_sample_indices: dict = defaultdict(list)

        self.db_conn        = None

        cache_dir = os.environ.get("DCASE2026_DATASET_CACHE_DIR", self.DEFAULT_CACHE_DIR)
        self.cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
        os.makedirs(self.cache_dir, exist_ok=True)
        fingerprint   = _dataset_cache_fingerprint(sequence_infos)
        self.db_path  = os.path.join(self.cache_dir, f"annotations_{fingerprint}.db")

        journal_path = self.db_path + "-journal"
        if os.path.exists(journal_path):
            print(
                f"[Dataset] Incomplete SQLite cache detected — rebuilding: {self.db_path}"
            )
            for stale_path in (journal_path, self.db_path):
                try:
                    os.remove(stale_path)
                except FileNotFoundError:
                    pass

        if os.path.exists(self.db_path):
            print(f"[Dataset] Cache hit  — loading index from {self.db_path}")
            self._load_index_from_db()
        else:
            print(f"[Dataset] Cache miss — building SQLite database from JSONs...")
            self._build_db(sequence_infos)

        if self.frames_per_epoch is None:
            print(f"[Dataset] frames_per_epoch=None → using all {len(self.all_samples)} frames")
        else:
            suffix = f" | fixed_seed={self.subset_seed}" if self.fixed_subset else ""
            print(f"[Dataset] frames_per_epoch={self.frames_per_epoch} → subsampling{suffix}")

        self.reset_epoch()

    def _load_teacher23_csv_dir(self, csv_dir: str) -> dict:
        if not csv_dir:
            return {}
        if not os.path.isdir(csv_dir):
            print(f"[Dataset] Teacher23 CSV dir missing — disabled: {csv_dir}")
            return {}

        by_seq_frame = defaultdict(lambda: defaultdict(list))
        n_rows = 0
        for path in sorted(glob.glob(os.path.join(csv_dir, "*.csv"))):
            seq = os.path.splitext(os.path.basename(path))[0]
            with open(path, newline="") as f:
                for row in csv.reader(f):
                    if len(row) < 6:
                        continue
                    try:
                        frame = int(float(row[0]))
                        cls = int(float(row[1]))
                        x, y, z = float(row[3]), float(row[4]), float(row[5])
                    except ValueError:
                        continue
                    if cls < 0:
                        continue
                    px, py = _cart_to_json_xy(x, y, z, self.img_w, self.img_h)
                    by_seq_frame[seq][frame].append((px, py, cls))
                    n_rows += 1

        out = {seq: dict(frames) for seq, frames in by_seq_frame.items()}
        print(
            f"[Dataset] Teacher23 positions loaded: {len(out)} sequences | "
            f"{n_rows} rows from {csv_dir}"
        )
        return out

    def _get_db_conn(self):
        if self.db_conn is None:
            import pathlib
            db_uri       = pathlib.Path(self.db_path).absolute().as_uri()
            self.db_conn = sqlite3.connect(f"{db_uri}?mode=ro", uri=True)
        return self.db_conn

    def _build_db(self, sequence_infos: list):
        from tqdm import tqdm

        tmp_path = self.db_path + ".tmp"
        for stale_path in (tmp_path, tmp_path + "-journal"):
            try:
                os.remove(stale_path)
            except FileNotFoundError:
                pass

        conn = sqlite3.connect(tmp_path)
        c    = conn.cursor()
        c.execute("PRAGMA temp_store=FILE")
        c.execute("CREATE TABLE IF NOT EXISTS annots (seq_id INTEGER, frame_idx INTEGER, data BLOB)")
        c.execute("CREATE TABLE IF NOT EXISTS seqs (seq_id INTEGER, seq_dir TEXT, seq_name TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS frame_classes (sample_idx INTEGER, category_id INTEGER)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_lookup ON annots (seq_id, frame_idx)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_fc ON frame_classes (category_id)")
        c.execute("CREATE TEMP TABLE IF NOT EXISTS frame_ann_tmp (frame_idx INTEGER, data BLOB)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_frame_ann_tmp ON frame_ann_tmp (frame_idx)")

        seq_id              = 0
        n_pos, n_neg, n_ann = 0, 0, 0

        for json_path, seq_dir, seq_name in tqdm(sequence_infos, desc="[Dataset] Building SQLite DB", leave=False):
            c.execute("INSERT INTO seqs VALUES (?, ?, ?)", (seq_id, seq_dir, seq_name))
            self.seq_map[seq_id] = (seq_dir, seq_name)

            c.execute("DELETE FROM frame_ann_tmp")
            annotated_ids = set()
            for ann in _iter_json_annotations(json_path):
                fi = int(ann["metadata_frame_index"])
                annotated_ids.add(fi)
                c.execute(
                    "INSERT INTO frame_ann_tmp VALUES (?, ?)",
                    (fi, pickle.dumps(ann, protocol=pickle.HIGHEST_PROTOCOL)),
                )

            disk_ids      = set(scan_available_frames(seq_dir, seq_name))
            all_ids       = sorted(annotated_ids | disk_ids)

            for fi in all_ids:
                rows       = c.execute(
                    "SELECT data FROM frame_ann_tmp WHERE frame_idx=?",
                    (fi,),
                ).fetchall()
                anns       = [pickle.loads(row[0]) for row in rows]
                sample_idx = len(self.all_samples)
                self.all_samples.append((seq_id, fi))
                c.execute(
                    "INSERT INTO annots VALUES (?, ?, ?)",
                    (seq_id, fi, pickle.dumps(anns, protocol=pickle.HIGHEST_PROTOCOL)),
                )

                for ann in anns:
                    cat_id = int(ann["category_id"])
                    c.execute("INSERT INTO frame_classes VALUES (?, ?)", (sample_idx, cat_id))
                    self.class_to_sample_indices[cat_id].append(sample_idx)

                if anns:
                    n_pos += 1
                    n_ann += len(anns)
                else:
                    n_neg += 1

            seq_id += 1
            conn.commit()

        conn.commit()
        conn.close()
        os.replace(tmp_path, self.db_path)
        print(f"[Dataset] SQLite DB built. {len(sequence_infos)} sequences | "
              f"{len(self.all_samples)} frames ({n_pos} pos / {n_neg} neg) | "
              f"{n_ann} annotations | {len(self.class_to_sample_indices)} classes indexed")

    def _load_index_from_db(self):
        from tqdm import tqdm

        conn = sqlite3.connect(self.db_path)
        c    = conn.cursor()

        for row in c.execute("SELECT seq_id, seq_dir, seq_name FROM seqs"):
            self.seq_map[row[0]] = (row[1], row[2])

        c.execute("SELECT COUNT(*) FROM annots")
        total_rows = c.fetchone()[0]

        c.execute("SELECT seq_id, frame_idx FROM annots")
        for row in tqdm(c, total=total_rows, desc="[Dataset] Loading Index from DB", leave=False):
            self.all_samples.append((row[0], row[1]))

        for row in c.execute("SELECT sample_idx, category_id FROM frame_classes"):
            self.class_to_sample_indices[row[1]].append(row[0])

        if self.class_to_sample_indices:
            n_pairs = sum(len(v) for v in self.class_to_sample_indices.values())
            print(f"[Dataset] Class index loaded: {len(self.class_to_sample_indices)} classes, "
                  f"{n_pairs} annotated frame-class pairs")

        conn.close()

    def clear_caches(self):
        self.frame_cache.clear()
        if self.acoustic_extractor is not None:
            self.acoustic_extractor.clear_cache()

    def __del__(self):
        db_conn = getattr(self, "db_conn", None)
        if db_conn is not None:
            db_conn.close()

    def load_frame_tensor(self, seq_dir: str, seq_name: str, frame_idx: int, warn_missing: bool = True) -> torch.Tensor:
        key = (seq_name, frame_idx)

        if key in self.frame_cache:
            tensor = self.frame_cache.pop(key)
            self.frame_cache[key] = tensor
            return tensor
 
        path = frame_path(seq_dir, seq_name, frame_idx)
        if os.path.isfile(path):
            if self.gpu_rgb_pipeline and self.acoustic_extractor is None:
                try:
                    rgb = read_image(path, mode=ImageReadMode.RGB)
                except RuntimeError as exc:
                    if "mmap" not in str(exc).lower():
                        raise
                    # torchvision uses mmap in read_file; PIL avoids rare mmap exhaustion.
                    with Image.open(path) as img:
                        arr = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
                    rgb = torch.from_numpy(arr).permute(2, 0, 1)
            else:
                img = Image.open(path).convert("RGB")
                if img.size != (self.img_w, self.img_h):
                    img = img.resize((self.img_w, self.img_h), Image.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0
                rgb = torch.from_numpy(arr).permute(2, 0, 1)
        else:
            if warn_missing:
                print(f"[WARN] Frame not found — using noise: {path}")
            rng = np.random.default_rng(abs(hash(seq_name)) % 2**31 + int(frame_idx))
            if self.gpu_rgb_pipeline and self.acoustic_extractor is None:
                rgb = torch.from_numpy(
                    rng.integers(0, 256, size=(3, self.img_h, self.img_w), dtype=np.uint8)
                )
            else:
                rgb = torch.from_numpy(rng.random((3, self.img_h, self.img_w), dtype=np.float32))

        if self.acoustic_extractor is None:
            tensor = rgb
        else:
            wav_path = wav_path_from_seq_dir(seq_dir, self.frames_base, self.mic_base)
            acoustic = self.acoustic_extractor.get_frame_bands(wav_path, frame_idx)
            tensor   = torch.cat([rgb, acoustic], dim=0)

        if self.cache_max_size > 0:
            self.frame_cache[key] = tensor
            if len(self.frame_cache) > self.cache_max_size:
                self.frame_cache.popitem(last=False)

        return tensor

    def build_frame_meta(self, seq_dir: str, seq_name: str, frame_idx: int) -> dict:
        return {
            "seq_dir": seq_dir,
            "seq_name": seq_name,
            "frame_idx": int(frame_idx),
            "wav_path": wav_path_from_seq_dir(seq_dir, self.frames_base, self.mic_base),
        }

    def _points_to_position_heatmap(self, points: list) -> torch.Tensor:
        if not points:
            return torch.zeros(self.img_h, self.img_w, dtype=torch.float32)

        yy, xx = np.mgrid[0:self.img_h, 0:self.img_w].astype(np.float32)
        sigma2 = self.teacher23_sigma_px * self.teacher23_sigma_px
        target = np.zeros((self.img_h, self.img_w), dtype=np.float32)
        for px, py, *_rest in points:
            dx = np.abs(xx - float(px))
            dx = np.minimum(dx, float(self.img_w) - dx)
            dy = yy - float(py)
            target = np.maximum(target, np.exp(-(dx * dx + dy * dy) / (2.0 * sigma2)).astype(np.float32))
        peak = float(target.max())
        if peak > 1e-8:
            target /= peak
        return torch.from_numpy(target)

    def build_teacher23_target(self, seq_name: str, frame_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        points = self.teacher23_by_seq_frame.get(seq_name, {}).get(int(frame_idx), [])
        if not points:
            return (
                torch.zeros(self.img_h, self.img_w, dtype=torch.float32),
                torch.zeros((), dtype=torch.bool),
            )
        return self._points_to_position_heatmap(points), torch.ones((), dtype=torch.bool)

    def load_frame_tensor_legacy(self, seq_dir: str, seq_name: str, frame_idx: int, warn_missing: bool = True) -> torch.Tensor:
        """
        Backwards-compat alias kept for notebooks / ad-hoc scripts that may have imported it.
        """
        return self.load_frame_tensor(seq_dir, seq_name, frame_idx, warn_missing=warn_missing)

    def precache_frames(self, seq_dir: str, seq_name: str, frame_indices: list, verbose: bool = True):
        indices = sorted(frame_indices)
        if len(indices) > self.cache_max_size:
            return

        if verbose:
            print(f"[INFO] Pre-caching {len(indices)} frames …")
        missing = sum(1 for fi in indices if not os.path.isfile(frame_path(seq_dir, seq_name, fi)))
        for fi in indices:
            self.load_frame_tensor(seq_dir, seq_name, fi, warn_missing=(missing < 10))

        if verbose:
            found  = len(indices) - missing
            status = "all found" if missing == 0 else f"{missing} replaced with noise"
            print(f"[INFO] Cache: {found}/{len(indices)} frames on disk ({status})")

    def build_annotation_target(self, frame_annots: list, seq_name: str | None = None, frame_idx: int | None = None) -> dict:
        boxes, labels       = [], []
        bin_masks           = []
        energy_maps_list    = []
        energy_masks_list   = []
        distances, iids     = [], []
        gt_peak_points      = []

        combined_energy = np.zeros((self.img_h, self.img_w), dtype=np.float32)
        combined_mask   = np.zeros((self.img_h, self.img_w), dtype=bool)

        for ann in frame_annots:
            cat  = int(ann["category_id"]) + 1
            dist = float(ann["distance"])
            iid  = int(ann["instance_id"])

            energy_map  = np.zeros((self.img_h, self.img_w), dtype=np.float32)
            energy_mask = np.zeros((self.img_h, self.img_w), dtype=bool)
            triplets = _segmentation_to_triplets(ann["segmentation"])
            if triplets.shape[0] == 0:
                continue
            peak_idx = int(np.argmax(triplets[:, 2]))
            gt_peak_points.append((float(triplets[peak_idx, 0]), float(triplets[peak_idx, 1]), cat - 1))

            if self.annotation_target_mode == "dense_eval":
                dense = _render_segmentation_dense(
                    ann["segmentation"],
                    self.img_w,
                    self.img_h,
                )
                energy_map = np.maximum(energy_map, dense)
                energy_mask = dense > 1e-6
                active = np.argwhere(energy_mask)
                if active.shape[0] == 0:
                    continue
                ys = active[:, 0].astype(np.int32, copy=False)
                xs = active[:, 1].astype(np.int32, copy=False)
            else:
                coords = np.rint(triplets[:, :2]).astype(np.int32, copy=False)
                values = triplets[:, 2].astype(np.float32, copy=False)

                valid = (
                    (coords[:, 0] >= 0) & (coords[:, 0] < self.img_w) &
                    (coords[:, 1] >= 0) & (coords[:, 1] < self.img_h)
                )
                if not np.any(valid):
                    continue

                xs = coords[valid, 0]
                ys = coords[valid, 1]
                vals = values[valid]
                energy_mask[ys, xs] = True
                np.maximum.at(energy_map, (ys, xs), vals)

            combined_mask[ys, xs] = True
            combined_energy = np.maximum(combined_energy, energy_map)

            x0 = int(xs.min())
            x1 = int(xs.max()) + 1   # exclusive right
            y0 = int(ys.min())
            y1 = int(ys.max()) + 1   # exclusive bottom

            boxes.append([float(x0), float(y0), float(x1), float(y1)])
            labels.append(cat)
            bin_masks.append(energy_mask.copy())
            energy_maps_list.append(energy_map.copy())
            energy_masks_list.append(energy_mask.copy())
            distances.append(dist / self.dist_norm)
            iids.append(iid)

        if not boxes:
            N = 0
            teacher23_map, teacher23_valid = self.build_teacher23_target(seq_name, frame_idx) if seq_name is not None else (
                torch.zeros(self.img_h, self.img_w, dtype=torch.float32),
                torch.zeros((), dtype=torch.bool),
            )
            if not bool(teacher23_valid) and self.teacher23_fallback_gt_peak and gt_peak_points:
                teacher23_map = self._points_to_position_heatmap(gt_peak_points)
                teacher23_valid = torch.ones((), dtype=torch.bool)
            return dict(
                boxes        = torch.zeros(N, 4,              dtype=torch.float32),
                labels       = torch.zeros(N,                 dtype=torch.int64),
                masks        = torch.zeros(N, self.img_h, self.img_w, dtype=torch.bool),
                energy_maps  = torch.zeros(N, self.img_h, self.img_w, dtype=torch.float32),
                energy_masks = torch.zeros(N, self.img_h, self.img_w, dtype=torch.bool),
                vmap         = torch.zeros(self.img_h, self.img_w,    dtype=torch.float32),
                vmask        = torch.zeros(self.img_h, self.img_w,    dtype=torch.bool),
                distances    = torch.zeros(N,                 dtype=torch.float32),
                instance_ids = torch.zeros(N,                 dtype=torch.int64),
                teacher23_map   = teacher23_map,
                teacher23_valid = teacher23_valid,
            )

        teacher23_map, teacher23_valid = self.build_teacher23_target(seq_name, frame_idx) if seq_name is not None else (
            torch.zeros(self.img_h, self.img_w, dtype=torch.float32),
            torch.zeros((), dtype=torch.bool),
        )
        if not bool(teacher23_valid) and self.teacher23_fallback_gt_peak and gt_peak_points:
            teacher23_map = self._points_to_position_heatmap(gt_peak_points)
            teacher23_valid = torch.ones((), dtype=torch.bool)
        return dict(
            boxes        = torch.tensor(boxes,                           dtype=torch.float32),
            labels       = torch.tensor(labels,                          dtype=torch.int64),
            masks        = torch.tensor(np.stack(bin_masks),             dtype=torch.bool),
            energy_maps  = torch.tensor(np.stack(energy_maps_list),      dtype=torch.float32),
            energy_masks = torch.tensor(np.stack(energy_masks_list),     dtype=torch.bool),
            vmap         = torch.tensor(combined_energy,                 dtype=torch.float32),
            vmask        = torch.tensor(combined_mask,                   dtype=torch.bool),
            distances    = torch.tensor(distances,                       dtype=torch.float32),
            instance_ids = torch.tensor(iids,                            dtype=torch.int64),
            teacher23_map   = teacher23_map,
            teacher23_valid = teacher23_valid,
        )

    def reset_epoch(self):
        total = len(self.all_samples)
        n     = min(self.frames_per_epoch, total) if self.frames_per_epoch is not None else total

        if self.fixed_subset and self._fixed_current_indices is not None:
            self.current_indices = self._fixed_current_indices.copy()
            return

        rng = np.random.default_rng(self.subset_seed) if self.fixed_subset else np.random.default_rng()

        use_balanced = (
            self.class_balanced_sampling
            and bool(self.class_to_sample_indices)
        )

        if not use_balanced:
            indices = np.arange(total)
            rng.shuffle(indices)
            self.current_indices = indices[:n]
            if self.fixed_subset:
                self._fixed_current_indices = self.current_indices.copy()
            return

        n_balanced = int(round(n * self.class_balanced_fraction))
        n_balanced = max(0, min(n_balanced, n))
        # Protect against n_uniform > total crashing replace=False
        n_uniform  = min(n - n_balanced, total)

        uniform_part  = rng.choice(total, n_uniform, replace=False)

        focus_classes = getattr(self, "class_balanced_classes", None)
        if focus_classes:
            classes = [
                cls for cls in focus_classes
                if cls in self.class_to_sample_indices and self.class_to_sample_indices[cls]
            ]
        else:
            classes = list(self.class_to_sample_indices.keys())
        if not classes:
            classes = list(self.class_to_sample_indices.keys())
        balanced_part = np.empty(n_balanced, dtype=np.int64)
        for i in range(n_balanced):
            cls              = classes[int(rng.integers(len(classes)))]
            pool             = self.class_to_sample_indices[cls]
            balanced_part[i] = pool[int(rng.integers(len(pool)))]

        self.current_indices = rng.permutation(
            np.concatenate([uniform_part, balanced_part])
        )
        if self.fixed_subset:
            self._fixed_current_indices = self.current_indices.copy()

    def __len__(self):
        return len(self.current_indices)

    def __getitem__(self, idx):
        real_idx          = self.current_indices[idx]
        seq_id, fi        = self.all_samples[real_idx]
        seq_dir, seq_name = self.seq_map[seq_id]

        c = self._get_db_conn().cursor()
        c.execute("SELECT data FROM annots WHERE seq_id=? AND frame_idx=?", (seq_id, fi))
        row  = c.fetchone()
        anns = pickle.loads(row[0]) if row and row[0] else []

        image  = self.load_frame_tensor(seq_dir, seq_name, fi)
        target = self.build_annotation_target(anns, seq_name=seq_name, frame_idx=fi)

        if self.augmentor is not None:
            image, target = self.augmentor(image, target)
            return image, target

        if self.return_meta or self.acoustic_extractor is None:
            return image, target, self.build_frame_meta(seq_dir, seq_name, fi)

        return image, target

def collate_fn(batch):
    if batch and len(batch[0]) == 3:
        return [b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch]
    return [b[0] for b in batch], [b[1] for b in batch]

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        # Only clear the RAM-heavy frame cache. Leave the acoustic cache intact
        # so persistent workers don't repeatedly wipe their precomputed features.
        if hasattr(dataset, "frame_cache"):
            dataset.frame_cache.clear()


# ── 3. MODEL COMPONENTS ───────────────────────────────────────────────────────

class EnergyMaskHead(nn.Module):
    """
    Per-RoI energy mask predictor.
    """
    def __init__(self, in_ch: int = 256, mid: int = 256, dropout: float = 0.2):
        super().__init__()
        layers = []
        ch = in_ch
        for i in range(4):
            layers += [
                nn.Conv2d(ch, mid, 3, padding=1, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),
            ]
            if i in (1, 3):
                layers.append(nn.Dropout2d(dropout))
            ch = mid
        layers += [
            nn.ConvTranspose2d(mid, mid, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1),
            nn.Sigmoid(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   


def soft_iou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.float().flatten(1)
    target = target.float().flatten(1).clamp(0.0, 1.0)
    valid = target.sum(dim=1) > eps
    if not valid.any():
        return pred.new_zeros(1).squeeze()

    pred = pred[valid]
    target = target[valid]
    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - inter
    return (1.0 - (inter + eps) / (union + eps)).mean()


def weighted_mask_bce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 4.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    if pred.ndim == target.ndim - 1:
        pred = pred.unsqueeze(1)
    if target.ndim == pred.ndim - 1:
        target = target.unsqueeze(1)
    pred = pred.float().clamp(eps, 1.0 - eps)
    target = target.float().clamp(0.0, 1.0)
    valid = target.flatten(1).sum(dim=1) > eps
    if not valid.any():
        return pred.new_zeros(1).squeeze()

    pred = pred[valid]
    target = target[valid]
    weight = 1.0 + target * (float(max(pos_weight, 1.0)) - 1.0)
    loss = -(target * pred.log() + (1.0 - target) * torch.log1p(-pred))
    return (loss * weight).sum() / weight.sum().clamp_min(eps)


def weighted_focal_bce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 4.0,
    gamma: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    if pred.ndim == target.ndim - 1:
        pred = pred.unsqueeze(1)
    if target.ndim == pred.ndim - 1:
        target = target.unsqueeze(1)
    pred = pred.float().clamp(eps, 1.0 - eps)
    target = target.float().clamp(0.0, 1.0)
    valid = target.flatten(1).sum(dim=1) > eps
    if not valid.any():
        return pred.new_zeros(1).squeeze()

    pred = pred[valid]
    target = target[valid]
    weight = 1.0 + target * (float(max(pos_weight, 1.0)) - 1.0)
    pt = torch.where(target > 0.5, pred, 1.0 - pred)
    loss = -(target * pred.log() + (1.0 - target) * torch.log1p(-pred))
    loss = loss * torch.pow(1.0 - pt, float(max(gamma, 0.0)))
    return (loss * weight).sum() / weight.sum().clamp_min(eps)


def tversky_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    eps: float = 1e-6,
) -> torch.Tensor:
    if pred.ndim == target.ndim - 1:
        pred = pred.unsqueeze(1)
    if target.ndim == pred.ndim - 1:
        target = target.unsqueeze(1)
    pred = pred.float().clamp(eps, 1.0 - eps)
    target = target.float().clamp(0.0, 1.0)
    valid = target.flatten(1).sum(dim=1) > eps
    if not valid.any():
        return pred.new_zeros(1).squeeze()

    pred = pred[valid].flatten(1)
    target = target[valid].flatten(1)
    true_pos = (pred * target).sum(dim=1)
    false_pos = (pred * (1.0 - target)).sum(dim=1)
    false_neg = ((1.0 - pred) * target).sum(dim=1)
    denom = true_pos + float(alpha) * false_pos + float(beta) * false_neg
    return (1.0 - (true_pos + eps) / (denom + eps)).mean()


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Gradient of the Lovasz extension w.r.t. sorted prediction errors."""
    p = gt_sorted.numel()
    if p == 0:
        return gt_sorted
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-6)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def binary_lovasz_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Binary Lovasz-Softmax/Jaccard surrogate on probabilities.
    This directly targets IoU-style ranking errors and is useful for the
    official soft-mask AP objective, while keeping checkpoint structure intact.
    """
    if pred.ndim == target.ndim - 1:
        pred = pred.unsqueeze(1)
    if target.ndim == pred.ndim - 1:
        target = target.unsqueeze(1)
    pred = pred.float().clamp(0.0, 1.0)
    target = (target.float() > 0.5).float()

    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    losses = []
    for probs, labels in zip(pred_flat, target_flat):
        if labels.sum() <= 0:
            continue
        errors = (labels - probs).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        labels_sorted = labels[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(labels_sorted)))
    if not losses:
        return pred.new_zeros(1).squeeze()
    return torch.stack(losses).mean()


def energy_distribution_loss(
    pred: torch.Tensor,
    target_energy: torch.Tensor,
    target_gamma: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    if pred.ndim == target_energy.ndim - 1:
        pred = pred.unsqueeze(1)
    if target_energy.ndim == pred.ndim - 1:
        target_energy = target_energy.unsqueeze(1)

    pred = pred.float().flatten(1).clamp_min(eps)
    target = target_energy.float().flatten(1).clamp_min(0.0)
    valid = target.sum(dim=1) > eps
    if not valid.any():
        return pred.new_zeros(1).squeeze()

    pred = pred[valid]
    target = target[valid]
    if target_gamma != 1.0:
        target = target.pow(float(max(target_gamma, eps)))
    target_dist = target / target.sum(dim=1, keepdim=True).clamp_min(eps)
    pred_dist = pred / pred.sum(dim=1, keepdim=True).clamp_min(eps)
    target_log = target_dist.clamp_min(eps).log()
    pred_log = pred_dist.clamp_min(eps).log()
    return (target_dist * (target_log - pred_log)).sum(dim=1).mean()


def soft_boundary_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    kernel_size: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Soft boundary-IoU loss using a differentiable morphological boundary.
    This targets high-IoU mask alignment without changing checkpoint shape.
    """
    if pred.ndim == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
    elif pred.ndim == 3:
        pred = pred.unsqueeze(1)
    if target.ndim == 2:
        target = target.unsqueeze(0).unsqueeze(0)
    elif target.ndim == 3:
        target = target.unsqueeze(1)
    if pred.ndim != 4 or target.ndim != 4:
        raise ValueError("soft_boundary_loss expects HW, NHW, or NCHW tensors.")

    pred = pred.float().clamp(0.0, 1.0)
    target = (target.float() > 0.5).float()
    valid = target.flatten(1).sum(dim=1) > eps
    if not valid.any():
        return pred.new_zeros(1).squeeze()

    pred = pred[valid]
    target = target[valid]
    k = int(max(3, kernel_size))
    if k % 2 == 0:
        k += 1
    pad = k // 2

    pred_dilate = F.max_pool2d(pred, kernel_size=k, stride=1, padding=pad)
    pred_erode = -F.max_pool2d(-pred, kernel_size=k, stride=1, padding=pad)
    target_dilate = F.max_pool2d(target, kernel_size=k, stride=1, padding=pad)
    target_erode = -F.max_pool2d(-target, kernel_size=k, stride=1, padding=pad)

    pred_boundary = (pred_dilate - pred_erode).clamp(0.0, 1.0).flatten(1)
    target_boundary = (target_dilate - target_erode).clamp(0.0, 1.0).flatten(1)
    inter = (pred_boundary * target_boundary).sum(dim=1)
    union = pred_boundary.sum(dim=1) + target_boundary.sum(dim=1) - inter
    return (1.0 - (inter + eps) / (union + eps)).mean()


class FPNEnergyDecoder(nn.Module):
    """
    FPN-based full-image energy map decoder.
    """
    def __init__(
        self,
        fpn_ch: int = 256,
        mid: int = 128,
        img_w: int = 360,
        img_h: int = 180,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.img_w  = img_w
        self.img_h  = img_h
        self.lat    = nn.ModuleList([nn.Conv2d(fpn_ch, mid, 1) for _ in range(4)])
        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mid, mid, 3, padding=1, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),
            )
            for _ in range(4)
        ])
        self.head = nn.Sequential(
            nn.Conv2d(mid, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, fpn: dict) -> torch.Tensor:
        feats = [fpn[str(i)] for i in range(4)]
        
        x = self.smooth[3](self.lat[3](feats[3]))
        for i in (2, 1, 0):
            x = F.interpolate(x, size=feats[i].shape[-2:], mode="nearest")
            x = self.smooth[i](x + self.lat[i](feats[i]))
            
        x = F.interpolate(x, size=(self.img_h, self.img_w), mode="bilinear", align_corners=False)
        return self.head(x)


class DistanceHead(nn.Module):
    """
    Per-RoI distance regressor.
    """
    def __init__(self, feat_ch: int = 256, dropout: float = 0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_ch, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.fc(x)).squeeze(-1)


def _prepare_foreground_vector(values, size: int, fill_value: float) -> torch.Tensor:
    if size <= 0:
        return torch.zeros(0, dtype=torch.float32)
    out = np.full(size, fill_value, dtype=np.float32)
    if values is not None:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        n_copy = min(size, arr.size)
        if n_copy > 0:
            out[:n_copy] = arr[:n_copy]
    return torch.from_numpy(out)


def apply_foreground_logit_adjustment(
    class_logits: torch.Tensor,
    fg_class_priors: torch.Tensor | None,
    tau: float,
) -> torch.Tensor:
    if tau == 0.0 or fg_class_priors is None or fg_class_priors.numel() == 0:
        return class_logits

    n_fg = min(int(fg_class_priors.numel()), max(class_logits.shape[1] - 1, 0))
    if n_fg <= 0:
        return class_logits

    priors = fg_class_priors[:n_fg].to(device=class_logits.device, dtype=class_logits.dtype)
    priors = priors.clamp(min=1e-6)
    offsets = -float(tau) * (priors.log() - priors.log().mean())

    adjusted = class_logits.clone()
    adjusted[:, 1 : 1 + n_fg] = adjusted[:, 1 : 1 + n_fg] + offsets.unsqueeze(0)
    return adjusted


def calibrate_inference_logits(
    class_logits: torch.Tensor,
    fg_class_priors: torch.Tensor | None,
    logit_adj_tau: float = 0.0,
    fg_bias: float = 0.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    calibrated = apply_foreground_logit_adjustment(class_logits, fg_class_priors, logit_adj_tau)
    if fg_bias != 0.0 and calibrated.shape[1] > 1:
        calibrated = calibrated.clone()
        calibrated[:, 1:] = calibrated[:, 1:] + float(fg_bias)

    temp = float(max(temperature, 1e-3))
    if temp != 1.0:
        calibrated = calibrated / temp
    return calibrated


def prior_aware_classification_loss(
    class_logits: torch.Tensor,
    labels: torch.Tensor | list,
    fg_class_priors: torch.Tensor | None = None,
    fg_class_weights: torch.Tensor | None = None,
    bg_loss_weight: float = 1.0,
    logit_adj_tau: float = 0.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    if isinstance(labels, (list, tuple)):
        if not labels:
            return class_logits.new_zeros(())
        labels = torch.cat(labels, dim=0)

    if labels.numel() == 0 or class_logits.shape[0] == 0:
        return class_logits.new_zeros(())

    adjusted_logits = apply_foreground_logit_adjustment(class_logits, fg_class_priors, logit_adj_tau)
    cls_loss_raw = F.cross_entropy(
        adjusted_logits,
        labels,
        reduction="none",
        label_smoothing=float(max(0.0, label_smoothing)),
    )

    class_weights = adjusted_logits.new_ones((adjusted_logits.shape[1],))
    class_weights[0] = float(max(bg_loss_weight, 1e-3))
    if fg_class_weights is not None and fg_class_weights.numel() > 0:
        n_fg = min(int(fg_class_weights.numel()), max(class_weights.numel() - 1, 0))
        fg_w = fg_class_weights[:n_fg].to(device=adjusted_logits.device, dtype=adjusted_logits.dtype)
        fg_mean = fg_w.mean().clamp(min=1e-6)
        class_weights[1 : 1 + n_fg] = fg_w / fg_mean

    sample_weights = class_weights.index_select(0, labels)
    return (cls_loss_raw * sample_weights).sum() / sample_weights.sum().clamp(min=1.0)


def prior_aware_fastrcnn_loss(
    class_logits: torch.Tensor,
    box_regression: torch.Tensor,
    labels: list,
    regression_targets: list,
    fg_class_priors: torch.Tensor | None = None,
    fg_class_weights: torch.Tensor | None = None,
    bg_loss_weight: float = 1.0,
    logit_adj_tau: float = 0.0,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    classification_loss = prior_aware_classification_loss(
        class_logits,
        labels,
        fg_class_priors=fg_class_priors,
        fg_class_weights=fg_class_weights,
        bg_loss_weight=bg_loss_weight,
        logit_adj_tau=logit_adj_tau,
        label_smoothing=label_smoothing,
    )

    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    num_samples, num_classes = class_logits.shape
    box_regression = box_regression.reshape(num_samples, box_regression.size(-1) // 4, 4)
    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()
    return classification_loss, box_loss


def add_image_class_prior_logits(
    class_logits: torch.Tensor,
    box_lists: list,
    class_prior_logits: torch.Tensor | None,
) -> torch.Tensor:
    if class_prior_logits is None or class_logits.numel() == 0:
        return class_logits
    if class_prior_logits.ndim == 1:
        class_prior_logits = class_prior_logits.unsqueeze(0)

    counts = [int(boxes.shape[0]) for boxes in box_lists]
    if sum(counts) != int(class_logits.shape[0]) or not counts:
        return class_logits

    class_prior_logits = class_prior_logits.to(
        device=class_logits.device,
        dtype=class_logits.dtype,
    )
    if class_prior_logits.shape[0] != len(counts):
        return class_logits

    expanded = class_prior_logits.repeat_interleave(
        torch.tensor(counts, device=class_logits.device),
        dim=0,
    )
    n_fg = min(int(expanded.shape[1]), max(int(class_logits.shape[1]) - 1, 0))
    if n_fg <= 0:
        return class_logits

    adjusted = class_logits.clone()
    adjusted[:, 1 : 1 + n_fg] = adjusted[:, 1 : 1 + n_fg] + expanded[:, :n_fg]
    return adjusted


class PriorAwareBoxRoIHeads(RoIHeads):
    def __init__(
        self,
        *args,
        fg_class_priors=None,
        fg_class_weights=None,
        bg_loss_weight: float = 1.0,
        logit_adj_tau: float = 0.0,
        infer_logit_adj_tau: float = 0.0,
        label_smoothing: float = 0.0,
        cls_loss_weight: float = 1.0,
        infer_score_temperature: float = 1.0,
        infer_fg_bias: float = 0.0,
        infer_score_power: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        n_fg = max(int(self.box_predictor.cls_score.out_features) - 1, 0)
        priors = _prepare_foreground_vector(fg_class_priors, n_fg, 1.0 / max(n_fg, 1))
        weights = _prepare_foreground_vector(fg_class_weights, n_fg, 1.0)
        if n_fg > 0:
            priors = priors / priors.sum().clamp(min=1e-6)
            weights = weights / weights.mean().clamp(min=1e-6)
        self.register_buffer("fg_class_priors", priors, persistent=False)
        self.register_buffer("fg_class_weights", weights, persistent=False)
        self.bg_loss_weight = float(bg_loss_weight)
        self.logit_adj_tau = float(logit_adj_tau)
        self.infer_logit_adj_tau = float(infer_logit_adj_tau)
        self.label_smoothing = float(label_smoothing)
        self.cls_loss_weight = float(max(cls_loss_weight, 0.0))
        self.infer_score_temperature = float(max(infer_score_temperature, 1e-3))
        self.infer_fg_bias = float(infer_fg_bias)
        self.infer_score_power = float(max(infer_score_power, 1e-3))

    def set_inference_calibration(
        self,
        *,
        infer_score_temperature: float | None = None,
        infer_fg_bias: float | None = None,
        infer_score_power: float | None = None,
    ) -> None:
        if infer_score_temperature is not None:
            self.infer_score_temperature = float(max(infer_score_temperature, 1e-3))
        if infer_fg_bias is not None:
            self.infer_fg_bias = float(infer_fg_bias)
        if infer_score_power is not None:
            self.infer_score_power = float(max(infer_score_power, 1e-3))

    def _calibrate_eval_logits(self, class_logits: torch.Tensor) -> torch.Tensor:
        return calibrate_inference_logits(
            class_logits,
            self.fg_class_priors,
            logit_adj_tau=self.infer_logit_adj_tau,
            fg_bias=self.infer_fg_bias,
            temperature=self.infer_score_temperature,
        )

    def _calibrate_detection_scores(self, scores: torch.Tensor) -> torch.Tensor:
        if self.infer_score_power == 1.0:
            return scores
        return scores.clamp(0.0, 1.0).pow(self.infer_score_power)

    def forward(
        self,
        features: dict,
        proposals: list,
        image_shapes: list,
        targets: list | None = None,
        audio_features: dict | None = None,
        class_prior_logits: torch.Tensor | None = None,
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                if t["boxes"].dtype not in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if t["labels"].dtype != torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            matched_idxs = None
            labels = None
            regression_targets = None

        class_logits, box_regression = self._compute_box_outputs(
            features,
            proposals,
            image_shapes,
            audio_features=audio_features,
            class_prior_logits=class_prior_logits,
        )

        result = []
        losses = {}
        if self.training:
            if labels is None or regression_targets is None:
                raise ValueError("labels and regression_targets must be available during training")
            loss_classifier, loss_box_reg = prior_aware_fastrcnn_loss(
                class_logits=class_logits,
                box_regression=box_regression,
                labels=labels,
                regression_targets=regression_targets,
                fg_class_priors=self.fg_class_priors,
                fg_class_weights=self.fg_class_weights,
                bg_loss_weight=self.bg_loss_weight,
                logit_adj_tau=self.logit_adj_tau,
                label_smoothing=self.label_smoothing,
            )
            loss_classifier = loss_classifier * self.cls_loss_weight
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            adjusted_logits = self._calibrate_eval_logits(class_logits)
            boxes, scores, labels = self.postprocess_detections(
                adjusted_logits, box_regression, proposals, image_shapes
            )
            for boxes_i, scores_i, labels_i in zip(boxes, scores, labels):
                result.append(
                    {
                        "boxes": boxes_i,
                        "labels": labels_i,
                        "scores": self._calibrate_detection_scores(scores_i),
                    }
                )

        return result, losses

    def _pool_box_features(
        self,
        features: dict,
        proposals: list,
        image_shapes: list,
        audio_features: dict | None = None,
        class_prior_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.box_roi_pool(features, proposals, image_shapes)

    def _compute_box_outputs(
        self,
        features: dict,
        proposals: list,
        image_shapes: list,
        audio_features: dict | None = None,
        class_prior_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        box_features = self._pool_box_features(
            features, proposals, image_shapes, audio_features=audio_features
        )
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)
        class_logits = add_image_class_prior_logits(
            class_logits,
            proposals,
            class_prior_logits,
        )
        return class_logits, box_regression

    def classification_loss_for_boxes(
        self,
        features: dict,
        box_lists: list,
        image_shapes: list,
        label_lists: list,
        audio_features: dict | None = None,
        class_prior_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = next(iter(features.values())).device
        if sum(int(b.shape[0]) for b in box_lists) <= 0:
            return torch.zeros((), device=device)

        class_logits, _ = self._compute_box_outputs(
            features,
            box_lists,
            image_shapes,
            audio_features=audio_features,
            class_prior_logits=class_prior_logits,
        )
        labels = [lbl.to(device=device, dtype=torch.int64) for lbl in label_lists]
        return prior_aware_classification_loss(
            class_logits,
            labels,
            fg_class_priors=self.fg_class_priors,
            fg_class_weights=self.fg_class_weights,
            bg_loss_weight=self.bg_loss_weight,
            logit_adj_tau=self.logit_adj_tau,
            label_smoothing=self.label_smoothing,
        )


class PriorAwareRoICMAFHeads(PriorAwareBoxRoIHeads):
    def __init__(
        self,
        *args,
        roi_fusion_head: nn.Module | None = None,
        use_roi_cmaf: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.roi_fusion_head = roi_fusion_head
        self.use_roi_cmaf = bool(use_roi_cmaf)

    def _pool_box_features(
        self,
        features: dict,
        proposals: list,
        image_shapes: list,
        audio_features: dict | None = None,
        class_prior_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        visual_box_features = self.box_roi_pool(features, proposals, image_shapes)
        if (
            not self.use_roi_cmaf
            or self.roi_fusion_head is None
            or audio_features is None
        ):
            return visual_box_features

        audio_box_features = self.box_roi_pool(audio_features, proposals, image_shapes)
        return self.roi_fusion_head(visual_box_features, audio_box_features)

    def _compute_box_outputs(
        self,
        features: dict,
        proposals: list,
        image_shapes: list,
        audio_features: dict | None = None,
        class_prior_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visual_box_features = self.box_roi_pool(features, proposals, image_shapes)
        visual_repr = self.box_head(visual_box_features)
        class_logits_visual, box_regression = self.box_predictor(visual_repr)

        if (
            not self.use_roi_cmaf
            or self.roi_fusion_head is None
            or audio_features is None
            or visual_box_features.numel() == 0
        ):
            class_logits_visual = add_image_class_prior_logits(
                class_logits_visual,
                proposals,
                class_prior_logits,
            )
            return class_logits_visual, box_regression

        audio_box_features = self.box_roi_pool(audio_features, proposals, image_shapes)
        fused_box_features = self.roi_fusion_head(visual_box_features, audio_box_features)
        fused_repr = self.box_head(fused_box_features)
        class_logits_fused, _ = self.box_predictor(fused_repr)

        # Keep bbox regression visual-only; audio only nudges the class logits.
        blend = 0.5 * torch.sigmoid(self.roi_fusion_head.scale_logit)
        class_logits = class_logits_visual + blend * (class_logits_fused - class_logits_visual)
        class_logits = add_image_class_prior_logits(
            class_logits,
            proposals,
            class_prior_logits,
        )
        return class_logits, box_regression


def build_prior_aware_roi_heads(
    base_roi_heads: RoIHeads,
    box_predictor: nn.Module,
    *,
    fg_class_priors=None,
    fg_class_weights=None,
    bg_loss_weight: float = 1.0,
    logit_adj_tau: float = 0.0,
    infer_logit_adj_tau: float = 0.0,
    label_smoothing: float = 0.0,
    cls_loss_weight: float = 1.0,
    infer_score_temperature: float = 1.0,
    infer_fg_bias: float = 0.0,
    infer_score_power: float = 1.0,
    roi_fusion_head: nn.Module | None = None,
    use_roi_cmaf: bool = False,
) -> PriorAwareRoICMAFHeads:
    return PriorAwareRoICMAFHeads(
        box_roi_pool=base_roi_heads.box_roi_pool,
        box_head=base_roi_heads.box_head,
        box_predictor=box_predictor,
        fg_iou_thresh=base_roi_heads.proposal_matcher.high_threshold,
        bg_iou_thresh=base_roi_heads.proposal_matcher.low_threshold,
        batch_size_per_image=base_roi_heads.fg_bg_sampler.batch_size_per_image,
        positive_fraction=base_roi_heads.fg_bg_sampler.positive_fraction,
        bbox_reg_weights=base_roi_heads.box_coder.weights,
        score_thresh=base_roi_heads.score_thresh,
        nms_thresh=base_roi_heads.nms_thresh,
        detections_per_img=base_roi_heads.detections_per_img,
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        keypoint_roi_pool=None,
        keypoint_head=None,
        keypoint_predictor=None,
        fg_class_priors=fg_class_priors,
        fg_class_weights=fg_class_weights,
        bg_loss_weight=bg_loss_weight,
        logit_adj_tau=logit_adj_tau,
        infer_logit_adj_tau=infer_logit_adj_tau,
        label_smoothing=label_smoothing,
        cls_loss_weight=cls_loss_weight,
        infer_score_temperature=infer_score_temperature,
        infer_fg_bias=infer_fg_bias,
        infer_score_power=infer_score_power,
        roi_fusion_head=roi_fusion_head,
        use_roi_cmaf=use_roi_cmaf,
    )


class EnergyInstanceModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        n_channels: int = 12,
        img_w: int = 360,
        img_h: int = 180,
        energy_annot_w: float = 5.0,
        fullmap_annot_w: float = 10.0,
        det_fusion_mode: str = "hybrid",
        fusion_scale_init: float = -4.0,
        roi_fg_class_priors=None,
        roi_fg_class_weights=None,
        roi_bg_loss_weight: float = 1.0,
        roi_logit_adj_tau: float = 0.0,
        roi_infer_logit_adj_tau: float = 0.0,
        roi_label_smoothing: float = 0.0,
        roi_cls_loss_weight: float = 1.0,
        roi_gt_aux_loss_weight: float = 0.0,
        energy_mask_loss_weight: float = 1.0,
        fullmap_loss_weight: float = 1.0,
        distance_loss_weight: float = 1.0,
        energy_mask_iou_w: float = 0.25,
        energy_mask_bce_w: float = 0.0,
        energy_mask_bce_pos_w: float = 4.0,
        energy_mask_target_mode: str = "support",
        energy_mask_core_thr: float = 0.5,
        energy_mask_focal_w: float = 0.0,
        energy_mask_focal_gamma: float = 2.0,
        energy_mask_tversky_w: float = 0.0,
        energy_mask_tversky_alpha: float = 0.3,
        energy_mask_tversky_beta: float = 0.7,
        energy_mask_lovasz_w: float = 0.0,
        energy_mask_boundary_w: float = 0.0,
        energy_mask_boundary_kernel: int = 3,
        energy_dist_w: float = 0.0,
        energy_dist_gamma: float = 1.0,
        fullmap_dist_w: float = 0.0,
        fullmap_dist_gamma: float = 1.0,
        fullmap_mask_iou_w: float = 0.0,
        fullmap_mask_bce_w: float = 0.0,
        fullmap_mask_bce_pos_w: float = 8.0,
        fullmap_mask_focal_w: float = 0.0,
        fullmap_mask_focal_gamma: float = 2.0,
        fullmap_mask_tversky_w: float = 0.0,
        fullmap_mask_tversky_alpha: float = 0.3,
        fullmap_mask_tversky_beta: float = 0.7,
        fullmap_mask_lovasz_w: float = 0.0,
        fullmap_mask_boundary_w: float = 0.0,
        fullmap_mask_boundary_kernel: int = 5,
        teacher23_pos_loss_w: float = 0.0,
        det_loss_weight: float = 1.0,
        gt_box_jitter_frac: float = 0.1,
        roi_infer_score_temperature: float = 1.0,
        roi_infer_fg_bias: float = 0.0,
        roi_infer_score_power: float = 1.0,
        use_roi_cmaf: bool = False,
        roi_cmaf_scale_init: float = -3.0,
        use_resnet_conformer: bool = False,
        resnet_conformer_layers: int = 2,
        resnet_conformer_ffn_dim: int = 1024,
        resnet_conformer_kernel_size: int = 31,
        resnet_conformer_dropout: float = 0.1,
        resnet_conformer_scale_init: float = -4.0,
        resnet_conformer_target: str = "roi",
        use_speech_prior: bool = False,
        speech_prior_dim: int = 0,
        speech_prior_hidden_dim: int = 512,
        speech_prior_dropout: float = 0.1,
        speech_prior_scale_init: float = -2.0,
        speech_prior_bundle: str = "WAVLM_BASE_PLUS",
        speech_prior_context_sec: float = 1.0,
    ):
        super().__init__()
        self.img_w           = img_w
        self.img_h           = img_h
        self.energy_annot_w  = energy_annot_w
        self.fullmap_annot_w = fullmap_annot_w
        self.n_channels      = n_channels
        self.n_rgb_channels  = 3
        self.n_audio_channels = max(0, n_channels - self.n_rgb_channels)
        if self.n_audio_channels <= 0:
            raise ValueError("EnergyInstanceModel expects at least one acoustic channel.")
        self.det_fusion_mode = str(det_fusion_mode).strip().lower()
        if self.det_fusion_mode not in {"visual", "fused", "hybrid"}:
            raise ValueError(
                f"Unsupported det_fusion_mode={det_fusion_mode!r}; "
                "expected 'visual', 'fused', or 'hybrid'."
            )
        self.fusion_scale_init = float(fusion_scale_init)
        self.roi_bg_loss_weight = float(roi_bg_loss_weight)
        self.roi_logit_adj_tau = float(roi_logit_adj_tau)
        self.roi_infer_logit_adj_tau = float(roi_infer_logit_adj_tau)
        self.roi_label_smoothing = float(roi_label_smoothing)
        self.roi_cls_loss_weight = float(max(roi_cls_loss_weight, 0.0))
        self.roi_gt_aux_loss_weight = float(max(roi_gt_aux_loss_weight, 0.0))
        self.energy_mask_loss_weight = float(max(energy_mask_loss_weight, 0.0))
        self.fullmap_loss_weight = float(max(fullmap_loss_weight, 0.0))
        self.distance_loss_weight = float(max(distance_loss_weight, 0.0))
        self.energy_mask_iou_w = float(max(energy_mask_iou_w, 0.0))
        self.energy_mask_bce_w = float(max(energy_mask_bce_w, 0.0))
        self.energy_mask_bce_pos_w = float(max(energy_mask_bce_pos_w, 1.0))
        self.energy_mask_target_mode = str(energy_mask_target_mode or "support").strip().lower()
        if self.energy_mask_target_mode not in {"support", "energy_soft", "energy_core"}:
            raise ValueError(
                f"Unsupported energy_mask_target_mode={energy_mask_target_mode!r}; "
                "expected 'support', 'energy_soft', or 'energy_core'."
            )
        self.energy_mask_core_thr = float(min(max(energy_mask_core_thr, 0.0), 1.0))
        self.energy_mask_focal_w = float(max(energy_mask_focal_w, 0.0))
        self.energy_mask_focal_gamma = float(max(energy_mask_focal_gamma, 0.0))
        self.energy_mask_tversky_w = float(max(energy_mask_tversky_w, 0.0))
        self.energy_mask_tversky_alpha = float(max(energy_mask_tversky_alpha, 0.0))
        self.energy_mask_tversky_beta = float(max(energy_mask_tversky_beta, 0.0))
        self.energy_mask_lovasz_w = float(max(energy_mask_lovasz_w, 0.0))
        self.energy_mask_boundary_w = float(max(energy_mask_boundary_w, 0.0))
        self.energy_mask_boundary_kernel = int(max(3, energy_mask_boundary_kernel))
        self.energy_dist_w = float(max(energy_dist_w, 0.0))
        self.energy_dist_gamma = float(max(energy_dist_gamma, 1e-3))
        self.fullmap_dist_w = float(max(fullmap_dist_w, 0.0))
        self.fullmap_dist_gamma = float(max(fullmap_dist_gamma, 1e-3))
        self.fullmap_mask_iou_w = float(max(fullmap_mask_iou_w, 0.0))
        self.fullmap_mask_bce_w = float(max(fullmap_mask_bce_w, 0.0))
        self.fullmap_mask_bce_pos_w = float(max(fullmap_mask_bce_pos_w, 1.0))
        self.fullmap_mask_focal_w = float(max(fullmap_mask_focal_w, 0.0))
        self.fullmap_mask_focal_gamma = float(max(fullmap_mask_focal_gamma, 0.0))
        self.fullmap_mask_tversky_w = float(max(fullmap_mask_tversky_w, 0.0))
        self.fullmap_mask_tversky_alpha = float(max(fullmap_mask_tversky_alpha, 0.0))
        self.fullmap_mask_tversky_beta = float(max(fullmap_mask_tversky_beta, 0.0))
        self.fullmap_mask_lovasz_w = float(max(fullmap_mask_lovasz_w, 0.0))
        self.fullmap_mask_boundary_w = float(max(fullmap_mask_boundary_w, 0.0))
        self.fullmap_mask_boundary_kernel = int(max(3, fullmap_mask_boundary_kernel))
        self.teacher23_pos_loss_w = float(max(teacher23_pos_loss_w, 0.0))
        self.det_loss_weight = float(max(det_loss_weight, 0.0))
        self.gt_box_jitter_frac = float(max(gt_box_jitter_frac, 0.0))
        self.roi_infer_score_temperature = float(max(roi_infer_score_temperature, 1e-3))
        self.roi_infer_fg_bias = float(roi_infer_fg_bias)
        self.roi_infer_score_power = float(max(roi_infer_score_power, 1e-3))
        self.use_roi_cmaf = bool(use_roi_cmaf)
        self.roi_cmaf_scale_init = float(roi_cmaf_scale_init)
        self.use_resnet_conformer = bool(use_resnet_conformer)
        self.resnet_conformer_layers = int(max(1, resnet_conformer_layers))
        self.resnet_conformer_ffn_dim = int(max(256, resnet_conformer_ffn_dim))
        self.resnet_conformer_kernel_size = int(max(3, resnet_conformer_kernel_size))
        if self.resnet_conformer_kernel_size % 2 == 0:
            self.resnet_conformer_kernel_size += 1
        self.resnet_conformer_dropout = float(max(0.0, resnet_conformer_dropout))
        self.resnet_conformer_scale_init = float(resnet_conformer_scale_init)
        self.resnet_conformer_target = str(resnet_conformer_target).strip().lower()
        if self.resnet_conformer_target not in {"roi", "task", "all"}:
            raise ValueError(
                f"Unsupported resnet_conformer_target={resnet_conformer_target!r}; "
                "expected 'roi', 'task', or 'all'."
            )
        self.use_speech_prior = bool(use_speech_prior)
        self.speech_prior_dim = int(max(0, speech_prior_dim))
        self.speech_prior_hidden_dim = int(max(16, speech_prior_hidden_dim))
        self.speech_prior_dropout = float(max(0.0, speech_prior_dropout))
        self.speech_prior_scale_init = float(speech_prior_scale_init)
        self.speech_prior_bundle = str(speech_prior_bundle)
        self.speech_prior_context_sec = float(max(speech_prior_context_sec, 0.1))
        if self.use_speech_prior and self.speech_prior_dim <= 0:
            raise ValueError("use_speech_prior=True requires speech_prior_dim > 0.")
        self.roi_fg_class_priors_cfg = (
            np.asarray(roi_fg_class_priors, dtype=np.float32).reshape(-1).tolist()
            if roi_fg_class_priors is not None else None
        )
        self.roi_fg_class_weights_cfg = (
            np.asarray(roi_fg_class_weights, dtype=np.float32).reshape(-1).tolist()
            if roi_fg_class_weights is not None else None
        )

        skip_pretrained = os.environ.get(
            "AV_SELD_SKIP_TORCHVISION_PRETRAINED", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        base_weights = None if skip_pretrained else MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        base = maskrcnn_resnet50_fpn(
            weights=base_weights,
            weights_backbone=None,
        )

        in_box = base.roi_heads.box_predictor.cls_score.in_features
        box_predictor = FastRCNNPredictor(in_box, num_classes)
        roi_fusion_head = (
            RoICMAFFusionHead(
                channels=256,
                nhead=8,
                dropout=0.1,
                initial_scale_logit=self.roi_cmaf_scale_init,
            )
            if self.use_roi_cmaf else None
        )
        base.roi_heads = build_prior_aware_roi_heads(
            base.roi_heads,
            box_predictor=box_predictor,
            fg_class_priors=self.roi_fg_class_priors_cfg,
            fg_class_weights=self.roi_fg_class_weights_cfg,
            bg_loss_weight=self.roi_bg_loss_weight,
            logit_adj_tau=self.roi_logit_adj_tau,
            infer_logit_adj_tau=self.roi_infer_logit_adj_tau,
            label_smoothing=self.roi_label_smoothing,
            cls_loss_weight=self.roi_cls_loss_weight,
            infer_score_temperature=self.roi_infer_score_temperature,
            infer_fg_bias=self.roi_infer_fg_bias,
            infer_score_power=self.roi_infer_score_power,
            roi_fusion_head=roi_fusion_head,
            use_roi_cmaf=self.use_roi_cmaf,
        )

        base.transform.min_size   = (min(img_h, img_w),)
        base.transform.max_size   = max(img_h, img_w) * 4
        base.transform.image_mean = [0.485, 0.456, 0.406] + [0.5]  * (n_channels - 3)
        base.transform.image_std  = [0.229, 0.224, 0.225] + [0.25] * (n_channels - 3)

        self.det             = base
        self.resnet_conformer = (
            ResNetConformerFPNAdapter(
                channels=256,
                token_hw=(3, 6),
                nhead=8,
                ffn_dim=self.resnet_conformer_ffn_dim,
                num_layers=self.resnet_conformer_layers,
                depthwise_conv_kernel_size=self.resnet_conformer_kernel_size,
                dropout=self.resnet_conformer_dropout,
                initial_scale_logit=self.resnet_conformer_scale_init,
            )
            if self.use_resnet_conformer else None
        )
        self.audio_backbone  = AcousticFeaturePyramid(
            in_channels=self.n_audio_channels,
            out_channels=256,
        )
        self.fpn_fusion      = CrossModalFPNFusion(
            channels=256,
            token_hw=(3, 6),
            nhead=8,
            dropout=0.1,
            initial_scale_logit=self.fusion_scale_init,
        )
        self.energy_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0","1","2","3"], output_size=14, sampling_ratio=2)
        self.energy_head     = EnergyMaskHead(in_ch=256, mid=256, dropout=0.2)
        self.energy_decoder  = FPNEnergyDecoder(fpn_ch=256, mid=128, img_w=img_w, img_h=img_h, dropout=0.1)
        self.distance_head   = DistanceHead(feat_ch=256, dropout=0.3)
        if self.use_speech_prior:
            self.speech_prior_adapter = nn.Sequential(
                nn.LayerNorm(self.speech_prior_dim),
                nn.Linear(self.speech_prior_dim, self.speech_prior_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.speech_prior_dropout),
                nn.Linear(self.speech_prior_hidden_dim, num_classes - 1),
            )
            nn.init.zeros_(self.speech_prior_adapter[-1].weight)
            nn.init.zeros_(self.speech_prior_adapter[-1].bias)
            self.speech_prior_scale_logit = nn.Parameter(
                torch.tensor(self.speech_prior_scale_init, dtype=torch.float32)
            )
        else:
            self.speech_prior_adapter = None
            self.speech_prior_scale_logit = None

    def _fpn4(self, features: dict) -> dict:
        return {k: features[k] for k in ("0","1","2","3") if k in features}

    def export_config(self) -> dict:
        return {
            "det_fusion_mode": self.det_fusion_mode,
            "fusion_scale_init": self.fusion_scale_init,
            "roi_fg_class_priors": self.roi_fg_class_priors_cfg,
            "roi_fg_class_weights": self.roi_fg_class_weights_cfg,
            "roi_bg_loss_weight": self.roi_bg_loss_weight,
            "roi_logit_adj_tau": self.roi_logit_adj_tau,
            "roi_infer_logit_adj_tau": self.roi_infer_logit_adj_tau,
            "roi_label_smoothing": self.roi_label_smoothing,
            "roi_cls_loss_weight": self.roi_cls_loss_weight,
            "roi_gt_aux_loss_weight": self.roi_gt_aux_loss_weight,
            "energy_mask_loss_weight": self.energy_mask_loss_weight,
            "fullmap_loss_weight": self.fullmap_loss_weight,
            "distance_loss_weight": self.distance_loss_weight,
            "energy_mask_iou_w": self.energy_mask_iou_w,
            "energy_mask_bce_w": self.energy_mask_bce_w,
            "energy_mask_bce_pos_w": self.energy_mask_bce_pos_w,
            "energy_mask_target_mode": self.energy_mask_target_mode,
            "energy_mask_core_thr": self.energy_mask_core_thr,
            "energy_mask_focal_w": self.energy_mask_focal_w,
            "energy_mask_focal_gamma": self.energy_mask_focal_gamma,
            "energy_mask_tversky_w": self.energy_mask_tversky_w,
            "energy_mask_tversky_alpha": self.energy_mask_tversky_alpha,
            "energy_mask_tversky_beta": self.energy_mask_tversky_beta,
            "energy_mask_lovasz_w": self.energy_mask_lovasz_w,
            "energy_mask_boundary_w": self.energy_mask_boundary_w,
            "energy_mask_boundary_kernel": self.energy_mask_boundary_kernel,
            "energy_dist_w": self.energy_dist_w,
            "energy_dist_gamma": self.energy_dist_gamma,
            "fullmap_dist_w": self.fullmap_dist_w,
            "fullmap_dist_gamma": self.fullmap_dist_gamma,
            "fullmap_mask_iou_w": self.fullmap_mask_iou_w,
            "fullmap_mask_bce_w": self.fullmap_mask_bce_w,
            "fullmap_mask_bce_pos_w": self.fullmap_mask_bce_pos_w,
            "fullmap_mask_focal_w": self.fullmap_mask_focal_w,
            "fullmap_mask_focal_gamma": self.fullmap_mask_focal_gamma,
            "fullmap_mask_tversky_w": self.fullmap_mask_tversky_w,
            "fullmap_mask_tversky_alpha": self.fullmap_mask_tversky_alpha,
            "fullmap_mask_tversky_beta": self.fullmap_mask_tversky_beta,
            "fullmap_mask_lovasz_w": self.fullmap_mask_lovasz_w,
            "fullmap_mask_boundary_w": self.fullmap_mask_boundary_w,
            "fullmap_mask_boundary_kernel": self.fullmap_mask_boundary_kernel,
            "teacher23_pos_loss_w": self.teacher23_pos_loss_w,
            "det_loss_weight": self.det_loss_weight,
            "gt_box_jitter_frac": self.gt_box_jitter_frac,
            "roi_infer_score_temperature": self.roi_infer_score_temperature,
            "roi_infer_fg_bias": self.roi_infer_fg_bias,
            "roi_infer_score_power": self.roi_infer_score_power,
            "use_roi_cmaf": self.use_roi_cmaf,
            "roi_cmaf_scale_init": self.roi_cmaf_scale_init,
            "use_resnet_conformer": self.use_resnet_conformer,
            "resnet_conformer_layers": self.resnet_conformer_layers,
            "resnet_conformer_ffn_dim": self.resnet_conformer_ffn_dim,
            "resnet_conformer_kernel_size": self.resnet_conformer_kernel_size,
            "resnet_conformer_dropout": self.resnet_conformer_dropout,
            "resnet_conformer_scale_init": self.resnet_conformer_scale_init,
            "resnet_conformer_target": self.resnet_conformer_target,
            "use_speech_prior": self.use_speech_prior,
            "speech_prior_dim": self.speech_prior_dim,
            "speech_prior_hidden_dim": self.speech_prior_hidden_dim,
            "speech_prior_dropout": self.speech_prior_dropout,
            "speech_prior_scale_init": self.speech_prior_scale_init,
            "speech_prior_bundle": self.speech_prior_bundle,
            "speech_prior_context_sec": self.speech_prior_context_sec,
        }

    def _speech_prior_logits(
        self,
        speech_prior_features,
        device: torch.device,
    ) -> torch.Tensor | None:
        if not self.use_speech_prior or self.speech_prior_adapter is None:
            return None
        if speech_prior_features is None:
            return None
        if isinstance(speech_prior_features, (list, tuple)):
            speech_prior_features = torch.stack(
                [x if isinstance(x, torch.Tensor) else torch.as_tensor(x) for x in speech_prior_features],
                dim=0,
            )
        if speech_prior_features.ndim == 1:
            speech_prior_features = speech_prior_features.unsqueeze(0)
        speech_prior_features = speech_prior_features.to(device=device, dtype=torch.float32)
        prior_logits = self.speech_prior_adapter(speech_prior_features)
        scale = torch.sigmoid(self.speech_prior_scale_logit)
        return prior_logits * scale

    @staticmethod
    def _jitter_boxes(
        boxes: torch.Tensor,
        img_h: int,
        img_w: int,
        jitter_frac: float = 0.1,
    ) -> torch.Tensor:
        if boxes.shape[0] == 0 or jitter_frac <= 0.0:
            return boxes
        w      = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0)
        h      = (boxes[:, 3] - boxes[:, 1]).clamp(min=1.0)
        scale  = torch.stack([w, h, w, h], dim=1) * jitter_frac
        delta  = (torch.rand_like(scale) * 2.0 - 1.0) * scale
        jittered        = boxes + delta
        jittered[:, 0] = jittered[:, 0].clamp(min=0.0, max=float(img_w) - 1.0)
        jittered[:, 1] = jittered[:, 1].clamp(min=0.0, max=float(img_h) - 1.0)
        jittered[:, 2] = torch.maximum(jittered[:, 2], jittered[:, 0] + 1.0).clamp_max(float(img_w))
        jittered[:, 3] = torch.maximum(jittered[:, 3], jittered[:, 1] + 1.0).clamp_max(float(img_h))
        return jittered

    def _roi_energy(self, features: dict, box_lists: list, image_sizes: list):
        M      = sum(len(b) for b in box_lists)
        device = next(iter(features.values())).device
        if M == 0:
            return (torch.zeros(0, 28, 28, device=device),
                    torch.zeros(0,         device=device))

        roi_feats   = self.energy_roi_pool(features, box_lists, image_sizes)
        energy_flat = self.energy_head(roi_feats).squeeze(1)
        dist_feat   = roi_feats.mean(dim=[-2, -1])
        dist_flat   = self.distance_head(dist_feat)
        return energy_flat, dist_flat

    def _gt_roi_energy_targets(
        self,
        targets: list,
        gt_boxes_t: list,
        img_sizes: list,
        orig_sizes: list,
        device: torch.device,
    ) -> tuple:
        e_list, support_list, mask_list = [], [], []

        for tgt, boxes_t, (th, tw), (oh, ow) in zip(
            targets, gt_boxes_t, img_sizes, orig_sizes
        ):
            emaps  = tgt["energy_maps"]    
            emasks = tgt["energy_masks"]   

            for n in range(len(boxes_t)):
                x0 = max(0,  int(math.floor(boxes_t[n, 0].item() * ow / tw)))
                y0 = max(0,  int(math.floor(boxes_t[n, 1].item() * oh / th)))
                x1 = min(ow, int(math.ceil (boxes_t[n, 2].item() * ow / tw)))
                y1 = min(oh, int(math.ceil (boxes_t[n, 3].item() * oh / th)))

                if x1 <= x0 or y1 <= y0:
                    e_list.append(torch.zeros(1, 1, 28, 28, device=device))
                    support_list.append(torch.zeros(1, 1, 28, 28, device=device))
                    mask_list.append(torch.zeros(1, 1, 28, 28, device=device))
                    continue

                crop_e = emaps[n, y0:y1, x0:x1].unsqueeze(0).unsqueeze(0).to(device)
                crop_m = emasks[n, y0:y1, x0:x1].float().unsqueeze(0).unsqueeze(0).to(device)
                r_e = F.interpolate(crop_e, (28, 28), mode="bilinear", align_corners=False)
                r_support = F.interpolate(crop_m, (28, 28), mode="bilinear", align_corners=False)
                r_mask = r_support
                if self.energy_mask_target_mode != "support":
                    max_e = r_e.flatten(1).amax(dim=1).view(-1, 1, 1, 1)
                    norm_e = torch.where(
                        max_e > 1e-6,
                        r_e / max_e.clamp_min(1e-6),
                        torch.zeros_like(r_e),
                    ).clamp(0.0, 1.0)
                    if self.energy_mask_target_mode == "energy_soft":
                        r_mask = norm_e
                    elif self.energy_mask_target_mode == "energy_core":
                        r_mask = (norm_e >= self.energy_mask_core_thr).float()
                e_list.append(r_e)
                support_list.append(r_support)
                mask_list.append(r_mask)

        if not e_list:
            return (torch.zeros(0, 1, 28, 28, device=device),
                    torch.zeros(0, 1, 28, 28, device=device),
                    torch.zeros(0, 1, 28, 28, device=device))
        return torch.cat(e_list), torch.cat(support_list), torch.cat(mask_list)

    def forward(self, images: list, targets: list = None, speech_prior_features=None):
        is_train   = self.training
        orig_sizes = [img.shape[-2:] for img in images]

        images_t, targets_t = self.det.transform(images, targets)
        img_sizes   = images_t.image_sizes
        class_prior_logits = self._speech_prior_logits(
            speech_prior_features,
            images_t.tensors.device,
        )

        rgb_tensors = images_t.tensors[:, :self.n_rgb_channels]
        acoustic_tensors = images_t.tensors[:, self.n_rgb_channels:]

        visual_features_raw = self.det.backbone(rgb_tensors)
        if self.resnet_conformer is not None:
            visual_features_ctx = self.resnet_conformer(visual_features_raw)
        else:
            visual_features_ctx = visual_features_raw

        audio_fpn       = self.audio_backbone(acoustic_tensors)
        fusion_visual_features = (
            visual_features_ctx
            if self.resnet_conformer_target in {"roi", "task", "all"}
            else visual_features_raw
        )
        fused_features  = self.fpn_fusion(fusion_visual_features, audio_fpn)
        if self.det_fusion_mode == "fused":
            rpn_features = fused_features
            roi_features = fused_features
        elif self.det_fusion_mode == "visual":
            rpn_features = visual_features_ctx if self.resnet_conformer_target == "all" else visual_features_raw
            roi_features = visual_features_ctx
        else:
            rpn_features = visual_features_ctx if self.resnet_conformer_target == "all" else visual_features_raw
            roi_features = fused_features
        if self.use_roi_cmaf:
            roi_det_features = (
                visual_features_ctx
                if self.resnet_conformer_target in {"roi", "all"}
                else visual_features_raw
            )
            roi_audio_features = audio_fpn
        else:
            roi_det_features = roi_features
            roi_audio_features = None
        task_features   = fused_features
        fpn4            = self._fpn4(task_features)
        full_energy = self.energy_decoder(fpn4)

        if is_train:
            device = images_t.tensors.device

            proposals, rpn_losses = self.det.rpn(images_t, rpn_features, targets_t)
            _,         roi_losses = self.det.roi_heads(
                roi_det_features,
                proposals,
                img_sizes,
                targets_t,
                audio_features=roi_audio_features,
                class_prior_logits=class_prior_logits,
            )

            gt_boxes_t = [
                self._jitter_boxes(tgt["boxes"].to(device), h, w, self.gt_box_jitter_frac)
                for tgt, (h, w) in zip(targets_t, img_sizes)
            ]
            gt_labels_t = [tgt["labels"].to(device) for tgt in targets_t]
            if self.roi_gt_aux_loss_weight > 0.0:
                gt_cls_aux_loss = self.det.roi_heads.classification_loss_for_boxes(
                    roi_det_features,
                    gt_boxes_t,
                    img_sizes,
                    gt_labels_t,
                    audio_features=roi_audio_features,
                    class_prior_logits=class_prior_logits,
                ) * self.roi_gt_aux_loss_weight
            energy_flat, dist_flat = self._roi_energy(fpn4, gt_boxes_t, img_sizes)

            gt_e, gt_support, gt_m = self._gt_roi_energy_targets(
                targets, gt_boxes_t, img_sizes, orig_sizes, device
            )
            M = energy_flat.shape[0]

            if M > 0:
                pred        = energy_flat.unsqueeze(1)
                loss_e_base = F.mse_loss(pred, gt_e)
                ann_px      = gt_support > 0.5
                if ann_px.any():
                    loss_e_ann = F.mse_loss(pred[ann_px], gt_e[ann_px])
                else:
                    loss_e_ann = pred.new_zeros(1).squeeze()
                energy_loss = loss_e_base + self.energy_annot_w * loss_e_ann
                if self.energy_mask_iou_w > 0.0:
                    energy_iou_loss = (
                        soft_iou_loss(pred.squeeze(1), gt_m.squeeze(1))
                        * self.energy_mask_iou_w
                    )
                else:
                    energy_iou_loss = pred.new_zeros(1).squeeze()
                if self.energy_mask_bce_w > 0.0:
                    energy_bce_loss = (
                        weighted_mask_bce_loss(
                            pred,
                            gt_m,
                            pos_weight=self.energy_mask_bce_pos_w,
                        )
                        * self.energy_mask_bce_w
                    )
                else:
                    energy_bce_loss = pred.new_zeros(1).squeeze()
                if self.energy_mask_focal_w > 0.0:
                    energy_focal_loss = (
                        weighted_focal_bce_loss(
                            pred,
                            gt_m,
                            pos_weight=self.energy_mask_bce_pos_w,
                            gamma=self.energy_mask_focal_gamma,
                        )
                        * self.energy_mask_focal_w
                    )
                else:
                    energy_focal_loss = pred.new_zeros(1).squeeze()
                if self.energy_mask_tversky_w > 0.0:
                    energy_tversky_loss = (
                        tversky_loss(
                            pred,
                            gt_m,
                            alpha=self.energy_mask_tversky_alpha,
                            beta=self.energy_mask_tversky_beta,
                        )
                        * self.energy_mask_tversky_w
                    )
                else:
                    energy_tversky_loss = pred.new_zeros(1).squeeze()
                if self.energy_mask_lovasz_w > 0.0:
                    energy_lovasz_loss = (
                        binary_lovasz_loss(pred, gt_m)
                        * self.energy_mask_lovasz_w
                    )
                else:
                    energy_lovasz_loss = pred.new_zeros(1).squeeze()
                if self.energy_mask_boundary_w > 0.0:
                    energy_boundary_loss = (
                        soft_boundary_loss(
                            pred,
                            gt_m,
                            kernel_size=self.energy_mask_boundary_kernel,
                        )
                        * self.energy_mask_boundary_w
                    )
                else:
                    energy_boundary_loss = pred.new_zeros(1).squeeze()
                if self.energy_dist_w > 0.0:
                    energy_dist_loss = (
                        energy_distribution_loss(
                            pred,
                            gt_e,
                            target_gamma=self.energy_dist_gamma,
                        )
                        * self.energy_dist_w
                    )
                else:
                    energy_dist_loss = pred.new_zeros(1).squeeze()
            else:
                energy_loss = images_t.tensors.new_zeros(1).squeeze()
                energy_iou_loss = images_t.tensors.new_zeros(1).squeeze()
                energy_bce_loss = images_t.tensors.new_zeros(1).squeeze()
                energy_focal_loss = images_t.tensors.new_zeros(1).squeeze()
                energy_tversky_loss = images_t.tensors.new_zeros(1).squeeze()
                energy_lovasz_loss = images_t.tensors.new_zeros(1).squeeze()
                energy_boundary_loss = images_t.tensors.new_zeros(1).squeeze()
                energy_dist_loss = images_t.tensors.new_zeros(1).squeeze()

            gt_dists = torch.cat([tgt["distances"].to(device) for tgt in targets])
            if len(gt_dists) > 0 and len(dist_flat) == len(gt_dists):
                dist_loss = F.smooth_l1_loss(dist_flat, gt_dists)
            else:
                dist_loss = images_t.tensors.new_zeros(1).squeeze()

            vmap_loss = images_t.tensors.new_zeros(1).squeeze()
            vmap_dist_loss = images_t.tensors.new_zeros(1).squeeze()
            vmap_iou_loss = images_t.tensors.new_zeros(1).squeeze()
            vmap_bce_loss = images_t.tensors.new_zeros(1).squeeze()
            vmap_focal_loss = images_t.tensors.new_zeros(1).squeeze()
            vmap_tversky_loss = images_t.tensors.new_zeros(1).squeeze()
            vmap_lovasz_loss = images_t.tensors.new_zeros(1).squeeze()
            vmap_boundary_loss = images_t.tensors.new_zeros(1).squeeze()
            teacher23_pos_loss = images_t.tensors.new_zeros(1).squeeze()
            teacher23_n = 0
            for i, tgt in enumerate(targets):
                pred_i = full_energy[i, 0]
                gt_v   = tgt["vmap"].to(device)
                vm     = tgt["vmask"].to(device)
                W      = torch.ones_like(pred_i)
                W[vm]  = self.fullmap_annot_w
                vmap_loss = vmap_loss + (W * (pred_i - gt_v).pow(2)).mean()
                if self.fullmap_dist_w > 0.0:
                    vmap_dist_loss = vmap_dist_loss + energy_distribution_loss(
                        pred_i.unsqueeze(0),
                        gt_v.unsqueeze(0),
                        target_gamma=self.fullmap_dist_gamma,
                    )
                if self.fullmap_mask_iou_w > 0.0:
                    vmap_iou_loss = vmap_iou_loss + soft_iou_loss(
                        pred_i.unsqueeze(0),
                        vm.float().unsqueeze(0),
                    )
                if self.fullmap_mask_bce_w > 0.0:
                    vmap_bce_loss = vmap_bce_loss + weighted_mask_bce_loss(
                        pred_i.unsqueeze(0),
                        vm.float().unsqueeze(0),
                        pos_weight=self.fullmap_mask_bce_pos_w,
                    )
                if self.fullmap_mask_focal_w > 0.0:
                    vmap_focal_loss = vmap_focal_loss + weighted_focal_bce_loss(
                        pred_i.unsqueeze(0),
                        vm.float().unsqueeze(0),
                        pos_weight=self.fullmap_mask_bce_pos_w,
                        gamma=self.fullmap_mask_focal_gamma,
                    )
                if self.fullmap_mask_tversky_w > 0.0:
                    vmap_tversky_loss = vmap_tversky_loss + tversky_loss(
                        pred_i.unsqueeze(0),
                        vm.float().unsqueeze(0),
                        alpha=self.fullmap_mask_tversky_alpha,
                        beta=self.fullmap_mask_tversky_beta,
                    )
                if self.fullmap_mask_lovasz_w > 0.0:
                    vmap_lovasz_loss = vmap_lovasz_loss + binary_lovasz_loss(
                        pred_i.unsqueeze(0),
                        vm.float().unsqueeze(0),
                    )
                if self.fullmap_mask_boundary_w > 0.0:
                    vmap_boundary_loss = vmap_boundary_loss + soft_boundary_loss(
                        pred_i.unsqueeze(0),
                        vm.float().unsqueeze(0),
                        kernel_size=self.fullmap_mask_boundary_kernel,
                    )
                if self.teacher23_pos_loss_w > 0.0 and bool(tgt.get("teacher23_valid", False)):
                    teacher_map = tgt["teacher23_map"].to(device).float()
                    valid = teacher_map > 1e-4
                    if valid.any():
                        teacher23_pos_loss = teacher23_pos_loss + F.mse_loss(
                            pred_i[valid],
                            teacher_map[valid],
                        )
                        teacher23_n += 1

            vmap_loss = vmap_loss / max(len(targets), 1)
            vmap_dist_loss = vmap_dist_loss / max(len(targets), 1)
            vmap_iou_loss = vmap_iou_loss / max(len(targets), 1)
            vmap_bce_loss = vmap_bce_loss / max(len(targets), 1)
            vmap_focal_loss = vmap_focal_loss / max(len(targets), 1)
            vmap_tversky_loss = vmap_tversky_loss / max(len(targets), 1)
            vmap_lovasz_loss = vmap_lovasz_loss / max(len(targets), 1)
            vmap_boundary_loss = vmap_boundary_loss / max(len(targets), 1)
            if teacher23_n > 0:
                teacher23_pos_loss = teacher23_pos_loss / teacher23_n

            losses = {
                "loss_energy_mask": energy_loss * self.energy_mask_loss_weight,
                "loss_energy_iou":  energy_iou_loss,
                "loss_fullmap":     vmap_loss * self.fullmap_loss_weight,
                "loss_distance":    dist_loss * self.distance_loss_weight,
            }
            if self.det_loss_weight > 0.0:
                for name, value in rpn_losses.items():
                    losses[name] = value * self.det_loss_weight
                for name, value in roi_losses.items():
                    losses[name] = value * self.det_loss_weight
            if self.energy_mask_bce_w > 0.0:
                losses["loss_mask_bce"] = energy_bce_loss
            if self.energy_mask_focal_w > 0.0:
                losses["loss_mask_focal"] = energy_focal_loss
            if self.energy_mask_tversky_w > 0.0:
                losses["loss_mask_tversky"] = energy_tversky_loss
            if self.energy_mask_lovasz_w > 0.0:
                losses["loss_mask_lovasz"] = energy_lovasz_loss
            if self.energy_mask_boundary_w > 0.0:
                losses["loss_mask_boundary"] = energy_boundary_loss
            if self.energy_dist_w > 0.0:
                losses["loss_energy_dist"] = energy_dist_loss
            if self.fullmap_dist_w > 0.0:
                losses["loss_fullmap_dist"] = vmap_dist_loss * self.fullmap_dist_w
            if self.fullmap_mask_iou_w > 0.0:
                losses["loss_fullmap_iou"] = vmap_iou_loss * self.fullmap_mask_iou_w
            if self.fullmap_mask_bce_w > 0.0:
                losses["loss_fullmap_bce"] = vmap_bce_loss * self.fullmap_mask_bce_w
            if self.fullmap_mask_focal_w > 0.0:
                losses["loss_fullmap_focal"] = vmap_focal_loss * self.fullmap_mask_focal_w
            if self.fullmap_mask_tversky_w > 0.0:
                losses["loss_fullmap_tversky"] = vmap_tversky_loss * self.fullmap_mask_tversky_w
            if self.fullmap_mask_lovasz_w > 0.0:
                losses["loss_fullmap_lovasz"] = vmap_lovasz_loss * self.fullmap_mask_lovasz_w
            if self.fullmap_mask_boundary_w > 0.0:
                losses["loss_fullmap_boundary"] = (
                    vmap_boundary_loss * self.fullmap_mask_boundary_w
                )
            if self.teacher23_pos_loss_w > 0.0:
                losses["loss_teacher23_pos"] = teacher23_pos_loss * self.teacher23_pos_loss_w
            if self.roi_gt_aux_loss_weight > 0.0:
                losses["loss_cls_gt_aux"] = gt_cls_aux_loss
            return losses

        else:
            proposals,        _ = self.det.rpn(images_t, rpn_features, None)
            detections_raw,   _ = self.det.roi_heads(
                roi_det_features,
                proposals,
                img_sizes,
                None,
                audio_features=roi_audio_features,
                class_prior_logits=class_prior_logits,
            )

            all_boxes = [raw["boxes"] for raw in detections_raw]
            counts    = [len(b) for b in all_boxes]

            if sum(counts) > 0:
                e_flat, d_flat = self._roi_energy(fpn4, all_boxes, img_sizes)
                e_per_img      = list(e_flat.split(counts))
                d_per_img      = list(d_flat.split(counts))
            else:
                e_per_img = [torch.zeros(0, 28, 28) for _ in detections_raw]
                d_per_img = [torch.zeros(0)         for _ in detections_raw]

            detections = self.det.transform.postprocess(detections_raw, img_sizes, orig_sizes)

            for i, det in enumerate(detections):
                det["energy_maps"] = e_per_img[i].detach().cpu()
                det["dist_pred"]   = d_per_img[i].detach().cpu()
                det["full_energy"] = full_energy[i, 0].detach().cpu()

            return detections


# ── 4. INSTANCE TRACKER ───────────────────────────────────────────────────────

class InstanceTracker:
    def __init__(self, iou_thr: float = 0.3, max_age: int = 5, img_w: int = 360, img_h: int = 180):
        self.iou_thr  = iou_thr
        self.max_age  = max_age
        self.img_w    = img_w
        self.img_h    = img_h
        self._nxt     = 0
        self.tracks: dict = {}

    def update(self, det: dict) -> list:
        boxes  = det.get("boxes",       torch.zeros(0, 4))
        labels = det.get("labels",      torch.zeros(0, dtype=torch.long))
        scores = det.get("scores",      torch.zeros(0))
        emaps  = det.get("energy_maps", torch.zeros(0, 28, 28))
        dists  = det.get("dist_pred",   torch.zeros(0))
        full_e = det.get("full_energy", torch.zeros(self.img_h, self.img_w))
        N      = len(boxes)

        def _track_to_result(tid: int, coasting: bool) -> dict:
            trk = self.tracks[tid]
            return dict(
                track_id    = tid,
                label       = trk["label"],
                # Preserve the last known score for coasting tracks
                score       = trk.get("last_score", 0.0) if coasting else None,
                box         = trk["box"].tolist(),
                energy_map  = trk["energy_map"],
                dist_pred   = trk["dist_pred"],
                full_energy = trk["full_energy"],
                coasting    = coasting,
            )

        if N == 0:
            results = []
            for tid in list(self.tracks.keys()):
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]
                else:
                    results.append(_track_to_result(tid, coasting=True))
            return results

        track_ids   = list(self.tracks.keys())
        track_boxes = (torch.stack([self.tracks[t]["box"] for t in track_ids])
                       if track_ids else torch.zeros(0, 4))

        iou_mat = (box_iou(track_boxes.cpu(), boxes.cpu()).numpy()
                   if track_boxes.numel() > 0 else np.zeros((0, N), dtype=np.float32))

        matched_det, matched_trk = {}, set()
        if iou_mat.size > 0:
            row_ind, col_ind = linear_sum_assignment(1.0 - iou_mat)
            for r, c in zip(row_ind, col_ind):
                if iou_mat[r, c] >= self.iou_thr:
                    tid = track_ids[r]
                    self.tracks[tid].update(
                        box         = boxes[c].cpu(),
                        age         = 0,
                        label       = int(labels[c]),
                        energy_map  = (emaps[c].cpu() if c < emaps.shape[0] else torch.zeros(28, 28)),
                        dist_pred   = (float(dists[c]) if c < dists.shape[0] else 0.0),
                        full_energy = full_e,
                        last_score  = float(scores[c]), # Cache last known score
                    )
                    self.tracks[tid]["hits"] += 1
                    matched_det[c] = tid
                    matched_trk.add(r)

        for c in range(N):
            if c not in matched_det:
                tid = self._nxt; self._nxt += 1
                self.tracks[tid] = dict(
                    box         = boxes[c].cpu(),
                    age         = 0, hits=1,
                    label       = int(labels[c]),
                    energy_map  = (emaps[c].cpu() if c < emaps.shape[0] else torch.zeros(28, 28)),
                    dist_pred   = (float(dists[c]) if c < dists.shape[0] else 0.0),
                    full_energy = full_e,
                    last_score  = float(scores[c]), # Cache last known score
                )
                matched_det[c] = tid

        for r, tid in enumerate(track_ids):
            if r not in matched_trk:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]

        results = []
        for c in range(N):
            tid = matched_det[c]
            r   = _track_to_result(tid, coasting=False)
            r["score"] = float(scores[c])
            results.append(r)

        for r_idx, tid in enumerate(track_ids):
            if r_idx not in matched_trk and tid in self.tracks:
                results.append(_track_to_result(tid, coasting=True))

        return results
