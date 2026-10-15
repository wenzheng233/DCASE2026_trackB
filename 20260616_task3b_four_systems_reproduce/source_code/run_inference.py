# ============================================================
#  ENERGY-FIELD INSTANCE SEGMENTATION  —  INFERENCE SCRIPT
#  run_inference.py
#
#  Usage:
#      python run_inference.py --exp_dir experiments/baseline_2026...
#      python run_inference.py --exp_dir path --score_thr 0.35
#      python run_inference.py --exp_dir path --split dev-test --num_seqs 5
#
#  Filtering pipeline (in order):
#    1. Score threshold       — drop low-confidence detections
#    2. Per-class NMS         — remove spatially overlapping boxes
#    3. Per-frame class cap   — at most N detections per class per frame
#    4. Track confirmation    — only export tracks seen >= min_hits frames
#    5. Energy sparsification — export top-K energy points per detection
#       from either the RoI energy head or the full-frame energy map.
# ============================================================

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import warnings
import random
from collections import defaultdict, OrderedDict

warnings.filterwarnings("ignore")

# Disable torch.compile / dynamo entirely — torchvision detection models
# (RPN anchor generator, RoI pooler) are not compile-compatible and produce
# floods of graph-break warnings with zero speedup benefit.
import torch._dynamo
torch._dynamo.disable()

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

# ── Resolve script directory so imports always work ──────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ── Import model classes only from model.py (no module-level constants there)
_model_spec = importlib.util.spec_from_file_location(
    "model", os.path.join(_SCRIPT_DIR, "model.py"))
_model_mod = importlib.util.module_from_spec(_model_spec)
_model_mod.__name__ = "model"
_model_spec.loader.exec_module(_model_mod)

EnergyInstanceModel   = _model_mod.EnergyInstanceModel
InstanceTracker       = _model_mod.InstanceTracker
get_sequence_infos    = _model_mod.get_sequence_infos
scan_available_frames = _model_mod.scan_available_frames
frame_path            = _model_mod.frame_path

from baseline_paths import FRAMES_BASE, LABELS_BASE, MIC_BASE, UPLAM_CHECKPOINT
from acoustic_features import AcousticFeatureExtractor, wav_path_from_seq_dir
from speech_prior import SpeechPriorExtractor

# ══════════════════════════════════════════════════════════════════════════════
#  MLOps UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
class Logger(object):
    """Routes stdout/stderr to both the console and a specified log file."""
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def set_seed(seed=42):
    """Ensures deterministic runs for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_class_id(value) -> int:
    if isinstance(value, int):
        cid = value
    else:
        raw = str(value).strip()
        if raw == "":
            raise ValueError("Empty class key is not allowed.")
        if raw.lstrip("+-").isdigit():
            cid = int(raw)
        else:
            slug = _slug_class_name(raw)
            if slug not in CLASS_NAME_TO_ID:
                raise ValueError(
                    f"Unknown class key '{value}'. Use class ids or names like 'Music'."
                )
            cid = CLASS_NAME_TO_ID[slug]
    if cid not in CLASS_NAMES:
        raise ValueError(f"Class id {cid} is out of range for export gating.")
    return cid


def parse_class_value_map(raw, *, value_type=float, arg_name="class override"):
    if raw is None or raw == "" or raw == {}:
        return {}

    if isinstance(raw, dict):
        parsed = raw
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = {}
            for chunk in text.split(","):
                item = chunk.strip()
                if not item:
                    continue
                if ":" in item:
                    key, value = item.split(":", 1)
                elif "=" in item:
                    key, value = item.split("=", 1)
                else:
                    raise ValueError(
                        f"Invalid {arg_name} entry '{item}'. Use key:value pairs or JSON."
                    )
                parsed[key.strip()] = value.strip()
    else:
        raise TypeError(f"{arg_name} must be a dict or string, got {type(raw).__name__}.")

    if not isinstance(parsed, dict):
        raise ValueError(f"{arg_name} must decode to a dict.")

    normalized = {}
    for key, value in parsed.items():
        cid = resolve_class_id(key)
        normalized[cid] = value_type(value)
    return normalized


def format_class_value_map(value_map, *, value_fmt=".2f"):
    if not value_map:
        return "off"
    parts = []
    for cid, value in sorted(value_map.items()):
        label = CLASS_NAMES.get(int(cid), f"class-{cid}")
        if isinstance(value, float):
            rendered = format(value, value_fmt)
        else:
            rendered = str(value)
        parts.append(f"{label}({cid})={rendered}")
    return ", ".join(parts)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS  (mirrors train.py — keep in sync if you change paths there)
# ══════════════════════════════════════════════════════════════════════════════

IMG_W            = 360
IMG_H            = 180
N_CHANNELS       = 12
N_ACOUSTIC       = 9
NUM_CLASSES      = 14
DIST_NORM        = 500.0
INFERENCE_HZ     = 10
SCORE_THR        = 0.05    
ENERGY_EXPORT_THR= 0.10
DEFAULT_ENERGY_TOP_K = 100
DEFAULT_MAX_JSON_MB = 19.5

COAST_DECAY      = 0.9

CLASS_NAMES = {
    0: "Female speech",
    1: "Male speech",
    2: "Clapping",
    3: "Telephone",
    4: "Laughter",
    5: "Domestic sounds",
    6: "Walk/footsteps",
    7: "Door open/close",
    8: "Music",
    9: "Musical instr.",
    10: "Water tap/shower",
    11: "Bell",
    12: "Knock",
}


def _slug_class_name(text: str) -> str:
    return " ".join(
        "".join(ch.lower() if ch.isalnum() else " " for ch in str(text)).split()
    )


CLASS_NAME_TO_ID = {}
for _cid, _name in CLASS_NAMES.items():
    CLASS_NAME_TO_ID[_slug_class_name(_name)] = _cid
CLASS_NAME_TO_ID.update({
    "musical instrument": 9,
    "musical instruments": 9,
    "walk footsteps": 6,
    "door": 7,
})

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE    = torch.device("cuda")
    _gpu_name = torch.cuda.get_device_name(0)
    torch.backends.cudnn.benchmark = True
else:
    DEVICE    = torch.device("cpu")
    _gpu_name = "CPU"

# ── Worker count: leave 1 core free, cap at 16 ───────────────────────────────
_CPU_COUNT  = os.cpu_count() or 1
NUM_WORKERS = min(max(_CPU_COUNT - 1, 0), 16)


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE DATASET
# ══════════════════════════════════════════════════════════════════════════════

class InferenceDataset(Dataset):
    def __init__(
        self,
        seq_dir          : str,
        seq_name         : str,
        frame_indices    : list,
        uplam_checkpoint : str = UPLAM_CHECKPOINT,
        n_acoustic       : int = N_ACOUSTIC,
        img_w            : int = IMG_W,
        img_h            : int = IMG_H,
        compute_acoustic : bool = True,
    ):
        self.seq_dir          = seq_dir
        self.seq_name         = seq_name
        self.frame_indices    = frame_indices
        self.uplam_checkpoint = uplam_checkpoint
        self.n_acoustic       = n_acoustic
        self.img_w            = img_w
        self.img_h            = img_h
        self.compute_acoustic = compute_acoustic
        self.wav_path         = wav_path_from_seq_dir(seq_dir, FRAMES_BASE, MIC_BASE)
        self._extractor       = None   

    def _ensure_extractor(self):
        if self._extractor is None:
            self._extractor = AcousticFeatureExtractor(
                uplam_checkpoint = self.uplam_checkpoint,
                device           = torch.device("cpu"),
                num_bands        = self.n_acoustic,
            )

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        fi   = self.frame_indices[idx]
        path = frame_path(self.seq_dir, self.seq_name, fi)

        if os.path.isfile(path):
            img = Image.open(path).convert("RGB")
            if img.size != (self.img_w, self.img_h):
                img = img.resize((self.img_w, self.img_h), Image.BILINEAR)
            rgb = torch.from_numpy(
                np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        else:
            rng = np.random.default_rng(
                abs(hash(self.seq_name)) % 2**31 + int(fi))
            rgb = torch.from_numpy(
                rng.random((3, self.img_h, self.img_w), dtype=np.float32))

        if not self.compute_acoustic:
            return fi, rgb

        self._ensure_extractor()
        acoustic = self._extractor.get_frame_bands(self.wav_path, fi)
        tensor   = torch.cat([rgb, acoustic], dim=0)   
        return fi, tensor


def _worker_init(worker_id):
    ds = torch.utils.data.get_worker_info().dataset
    if not getattr(ds, "compute_acoustic", True):
        return
    ds._extractor = AcousticFeatureExtractor(
        uplam_checkpoint = ds.uplam_checkpoint,
        device           = torch.device("cpu"),
        num_bands        = ds.n_acoustic,
    )


def _collate(batch):
    fis     = [x[0] for x in batch]
    tensors = [x[1] for x in batch]
    return fis, tensors


def build_loader(seq_dir, seq_name, frame_indices, batch_size, num_workers,
                 compute_acoustic=True):
    ds = InferenceDataset(seq_dir=seq_dir, seq_name=seq_name,
                          frame_indices=frame_indices,
                          compute_acoustic=compute_acoustic)
    return DataLoader(
        ds,
        batch_size         = batch_size,
        shuffle            = False,
        num_workers        = num_workers,
        pin_memory         = torch.cuda.is_available(),
        prefetch_factor    = 4 if num_workers > 0 else None,
        worker_init_fn     = _worker_init if num_workers > 0 else None,
        persistent_workers = num_workers > 0,
        drop_last          = False,
        collate_fn         = _collate,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run trained EnergyInstanceModel on all test sequences.")

    p.add_argument("--exp_dir",    type=str, required=True,
                   help="Path to the experiment directory (e.g., experiments/run_name_timestamp)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Specific checkpoint to load. Defaults to energy_seg_best.pth in exp_dir")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Prediction JSON directory. Defaults to exp_dir/inference_outputs.")
    p.add_argument("--split",      default="test")
    p.add_argument("--num_seqs",   type=int, default=None,
                   help="Number of sequences to randomly sample for inference.")
    p.add_argument("--seqs", type=str, default="",
                   help="Comma-separated sequence names to run, overriding --num_seqs.")
    p.add_argument("--seed",       type=int, default=42,
                   help="Random seed for sampling sequences.")

    p.add_argument("--score_thr",          type=float, default=0.35)
    p.add_argument("--nms_iou_thr",        type=float, default=0.30)
    p.add_argument("--max_dets_per_class", type=int,   default=3)
    p.add_argument("--min_hits",           type=int,   default=2)
    p.add_argument("--track_mean_score_thr", type=float, default=0.0,
                   help="Export only tracks whose mean non-coasting score reaches this threshold.")
    p.add_argument("--track_peak_score_thr", type=float, default=0.0,
                   help="Export only tracks whose peak non-coasting score reaches this threshold.")
    p.add_argument("--track_mean_score_thr_by_class", type=str, default="",
                   help="Optional class-specific mean-score overrides, e.g. 'Music:0.20,1:0.35'.")
    p.add_argument("--track_peak_score_thr_by_class", type=str, default="",
                   help="Optional class-specific peak-score overrides, e.g. '{\"Music\": 0.35}'.")
    p.add_argument("--max_total_dets_per_frame", type=int, default=0,
                   help="Optional global per-frame export cap after track gating (0 disables).")
    p.add_argument("--max_total_dets_per_frame_by_class", type=str, default="",
                   help="Optional class-specific export caps after track gating, e.g. 'Music:2,1:1'.")
    p.add_argument("--energy_top_k",       type=int,   default=DEFAULT_ENERGY_TOP_K)
    p.add_argument("--energy_export_thr",  type=float, default=ENERGY_EXPORT_THR)
    p.add_argument(
        "--energy_export_mode",
        choices=("topk", "region", "hull"),
        default="topk",
        help=(
            "How to serialise each predicted energy field. 'topk' exports the "
            "old sparse peak points; 'region' exports thresholded region "
            "boundary points plus high-energy interior points; 'hull' exports "
            "the convex hull of the active energy region plus high-energy "
            "interior points so the evaluator's linear interpolation renders "
            "a filled energy area."
        ),
    )
    p.add_argument(
        "--energy_region_rel_thr",
        type=float,
        default=0.35,
        help=(
            "Relative threshold used by --energy_export_mode region/hull. "
            "The active region threshold is max(energy_export_thr, "
            "peak * energy_region_rel_thr)."
        ),
    )
    p.add_argument(
        "--energy_region_pad",
        type=int,
        default=0,
        help=(
            "Optional dilation radius in energy-map cells for region/hull "
            "export. 0 keeps the raw connected energy region."
        ),
    )
    p.add_argument(
        "--max_json_mb",
        type=float,
        default=DEFAULT_MAX_JSON_MB,
        help=(
            "Maximum size for each prediction JSON in MiB. If compact output "
            "still exceeds this size, lowest-score annotations are removed "
            "until the file fits. Set <=0 to disable."
        ),
    )
    p.add_argument(
        "--energy_source",
        choices=("roi", "fullmap_box"),
        default="roi",
        help=(
            "Energy field used for exported segmentation points. 'roi' keeps "
            "the original RoI head; 'fullmap_box' crops the full-frame energy "
            "map inside each detection box."
        ),
    )

    p.add_argument("--iou_thr",     type=float, default=0.3)
    p.add_argument("--max_age",     type=int,   default=5)
    p.add_argument("--coast_decay", type=float, default=COAST_DECAY,
                   help="Confidence decay per coasting frame (default 0.9).")

    p.add_argument("--batch_size",  type=int, default=32,
                   help="Frames per forward pass (default 32 for A100-40GB).")
    p.add_argument("--num_workers", type=int, default=NUM_WORKERS,
                   help=f"DataLoader workers (default {NUM_WORKERS}).")
    p.add_argument("--num_classes", type=int, default=NUM_CLASSES)
    p.add_argument("--roi_infer_score_temperature", type=float, default=None,
                   help="Override ROI score temperature calibration from checkpoint config.")
    p.add_argument("--roi_infer_fg_bias", type=float, default=None,
                   help="Override foreground logit bias applied at ROI inference time.")
    p.add_argument("--roi_infer_score_power", type=float, default=None,
                   help="Override post-softmax score power for ROI detections.")
    p.add_argument("--gpu_acoustic", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="Compute UpLAM acoustic features batched on the inference GPU.")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def find_model_config_path(checkpoint_path: str) -> str | None:
    checkpoint_dir = os.path.dirname(checkpoint_path)
    parent_dir = os.path.dirname(checkpoint_dir)
    candidates = [
        os.path.join(checkpoint_dir, "model_config.json"),
        os.path.join(parent_dir, "model_config.json"),
        os.path.join(parent_dir, "configs", "model_config.json"),
    ]
    for config_path in candidates:
        if os.path.isfile(config_path):
            return config_path
    return None


def load_model(
    checkpoint_path: str,
    num_classes: int,
    *,
    roi_infer_score_temperature: float | None = None,
    roi_infer_fg_bias: float | None = None,
    roi_infer_score_power: float | None = None,
) -> EnergyInstanceModel:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[INFO] Loading checkpoint : {checkpoint_path}")
    config_path = find_model_config_path(checkpoint_path)
    model_kwargs = {}
    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as fh:
            model_kwargs = json.load(fh)
        print(
            "[INFO] Model config    : "
            f"det_fusion_mode={model_kwargs.get('det_fusion_mode', 'fused')} | "
            f"fusion_scale_init={model_kwargs.get('fusion_scale_init', 'n/a')} | "
            f"use_roi_cmaf={model_kwargs.get('use_roi_cmaf', False)} | "
            f"use_resnet_conformer={model_kwargs.get('use_resnet_conformer', False)} | "
            f"roi_gt_aux_w={model_kwargs.get('roi_gt_aux_loss_weight', 0.0)} | "
            f"energy_iou_w={model_kwargs.get('energy_mask_iou_w', 0.25)} | "
            f"roi_infer_tau={model_kwargs.get('roi_infer_logit_adj_tau', 0.0)} | "
            f"roi_score_temp={model_kwargs.get('roi_infer_score_temperature', 1.0)} | "
            f"roi_fg_bias={model_kwargs.get('roi_infer_fg_bias', 0.0)} | "
            f"roi_score_power={model_kwargs.get('roi_infer_score_power', 1.0)} | "
            f"speech_prior={model_kwargs.get('use_speech_prior', False)}"
        )
    model = EnergyInstanceModel(
        num_classes=num_classes,
        det_fusion_mode=model_kwargs.get("det_fusion_mode", "fused"),
        fusion_scale_init=model_kwargs.get("fusion_scale_init", -4.0),
        roi_fg_class_priors=model_kwargs.get("roi_fg_class_priors"),
        roi_fg_class_weights=model_kwargs.get("roi_fg_class_weights"),
        roi_bg_loss_weight=model_kwargs.get("roi_bg_loss_weight", 1.0),
        roi_logit_adj_tau=model_kwargs.get("roi_logit_adj_tau", 0.0),
        roi_infer_logit_adj_tau=model_kwargs.get("roi_infer_logit_adj_tau", 0.0),
        roi_label_smoothing=model_kwargs.get("roi_label_smoothing", 0.0),
        roi_gt_aux_loss_weight=model_kwargs.get("roi_gt_aux_loss_weight", 0.0),
        energy_mask_iou_w=model_kwargs.get("energy_mask_iou_w", 0.25),
        roi_infer_score_temperature=model_kwargs.get("roi_infer_score_temperature", 1.0),
        roi_infer_fg_bias=model_kwargs.get("roi_infer_fg_bias", 0.0),
        roi_infer_score_power=model_kwargs.get("roi_infer_score_power", 1.0),
        use_roi_cmaf=model_kwargs.get("use_roi_cmaf", False),
        roi_cmaf_scale_init=model_kwargs.get("roi_cmaf_scale_init", -3.0),
        use_resnet_conformer=model_kwargs.get("use_resnet_conformer", False),
        resnet_conformer_layers=model_kwargs.get("resnet_conformer_layers", 2),
        resnet_conformer_ffn_dim=model_kwargs.get("resnet_conformer_ffn_dim", 1024),
        resnet_conformer_kernel_size=model_kwargs.get("resnet_conformer_kernel_size", 31),
        resnet_conformer_dropout=model_kwargs.get("resnet_conformer_dropout", 0.1),
        resnet_conformer_scale_init=model_kwargs.get("resnet_conformer_scale_init", -4.0),
        resnet_conformer_target=model_kwargs.get("resnet_conformer_target", "roi"),
        use_speech_prior=model_kwargs.get("use_speech_prior", False),
        speech_prior_dim=model_kwargs.get("speech_prior_dim", 0),
        speech_prior_hidden_dim=model_kwargs.get("speech_prior_hidden_dim", 512),
        speech_prior_dropout=model_kwargs.get("speech_prior_dropout", 0.1),
        speech_prior_scale_init=model_kwargs.get("speech_prior_scale_init", -2.0),
        speech_prior_bundle=model_kwargs.get("speech_prior_bundle", "WAVLM_BASE_PLUS"),
        speech_prior_context_sec=model_kwargs.get("speech_prior_context_sec", 1.0),
    )
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing:    print(f"[WARN] Missing keys    : {missing}")
    if unexpected: print(f"[WARN] Unexpected keys: {unexpected}")

    model.to(DEVICE)
    model.eval()
    roi_heads = getattr(model.det, "roi_heads", None)
    if roi_heads is not None and hasattr(roi_heads, "set_inference_calibration"):
        roi_heads.set_inference_calibration(
            infer_score_temperature=roi_infer_score_temperature,
            infer_fg_bias=roi_infer_fg_bias,
            infer_score_power=roi_infer_score_power,
        )
        print(
            "[INFO] ROI score calibration: "
            f"temp={getattr(roi_heads, 'infer_score_temperature', 1.0):.3f} | "
            f"fg_bias={getattr(roi_heads, 'infer_fg_bias', 0.0):.3f} | "
            f"score_power={getattr(roi_heads, 'infer_score_power', 1.0):.3f}"
        )
    print("[INFO] Model ready in eval() mode.")
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-STAGE FILTERING
# ══════════════════════════════════════════════════════════════════════════════

def filter_detections(det, score_thr, nms_iou_thr, max_dets_per_class):
    boxes       = det.get("boxes",       torch.zeros(0, 4))
    labels      = det.get("labels",      torch.zeros(0, dtype=torch.long))
    scores      = det.get("scores",      torch.zeros(0))
    energy_maps = det.get("energy_maps", torch.zeros(0, 28, 28))
    dist_pred   = det.get("dist_pred",   torch.zeros(0))
    N           = scores.shape[0]

    _empty = {**det,
              "boxes": torch.zeros(0, 4),
              "labels": torch.zeros(0, dtype=torch.long),
              "scores": torch.zeros(0),
              "energy_maps": torch.zeros(0, 28, 28),
              "dist_pred": torch.zeros(0)}

    if N == 0:
        return _empty

    keep = scores >= score_thr
    if not keep.any():
        return _empty

    def sel(t, mask):
        return t[mask] if isinstance(t, torch.Tensor) and t.shape[0] == N else t

    boxes, labels, scores = sel(boxes, keep), sel(labels, keep), sel(scores, keep)
    energy_maps = sel(energy_maps, keep)
    dist_pred   = sel(dist_pred, keep)
    N = scores.shape[0]

    keep_nms = []
    for cls in labels.unique():
        m   = labels == cls
        idx = m.nonzero(as_tuple=True)[0]
        keep_nms.append(idx[nms(boxes[idx], scores[idx], nms_iou_thr)])
    if not keep_nms:
        return _empty
    ki = torch.cat(keep_nms)
    ki = ki[scores[ki].argsort(descending=True)]

    def idx_sel(t):
        return t[ki] if isinstance(t, torch.Tensor) and t.shape[0] >= ki.max() + 1 else t

    boxes, labels, scores = idx_sel(boxes), idx_sel(labels), idx_sel(scores)
    energy_maps, dist_pred = idx_sel(energy_maps), idx_sel(dist_pred)

    keep_cap = []
    for cls in labels.unique():
        idx = (labels == cls).nonzero(as_tuple=True)[0]
        keep_cap.append(idx[:max_dets_per_class])
    if not keep_cap:
        return _empty
    ki2 = torch.cat(keep_cap)

    def idx_sel2(t):
        return t[ki2] if isinstance(t, torch.Tensor) and t.shape[0] >= ki2.max() + 1 else t

    return {**det,
            "boxes": idx_sel2(boxes), "labels": idx_sel2(labels),
            "scores": idx_sel2(scores), "energy_maps": idx_sel2(energy_maps),
            "dist_pred": idx_sel2(dist_pred)}


def sparsify_energy_map(emap, box, top_k, thr):
    x0, y0, x1, y1 = box
    flat  = emap.flatten()
    n_pts = flat.shape[0]

    top_idx = (np.argpartition(flat, -top_k)[-top_k:]
               if 0 < top_k < n_pts else np.arange(n_pts))
    top_idx = top_idx[flat[top_idx] >= thr]

    triplets = []
    for i in top_idx:
        rj, ci = i // 28, i % 28
        # Sub-pixel accurate coordinate extraction for 28x28 RoI features.
        # This completely avoids the exclusive-box out-of-bounds overshoot error
        # while perfectly centering the bins mathematically.
        triplets.append([
            round(x0 + (ci + 0.5) / 28.0 * (x1 - x0), 4),
            round(y0 + (rj + 0.5) / 28.0 * (y1 - y0), 4),
            round(float(flat[i]), 6),
        ])

    if not triplets:
        pk = int(flat.argmax())
        rj, ci = pk // 28, pk % 28
        triplets.append([
            round(x0 + (ci + 0.5) / 28.0 * (x1 - x0), 4),
            round(y0 + (rj + 0.5) / 28.0 * (y1 - y0), 4),
            round(float(flat[pk]), 6),
        ])

    triplets.sort(key=lambda t: -t[2])
    return triplets


def _dilate_mask(mask, radius):
    mask = np.asarray(mask, dtype=bool)
    radius = int(max(radius, 0))
    if radius <= 0 or not mask.any():
        return mask

    out = mask.copy()
    for _ in range(radius):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        grown = np.zeros_like(out)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                grown |= padded[1 + dy:1 + dy + out.shape[0],
                                1 + dx:1 + dx + out.shape[1]]
        out = grown
    return out


def _peak_connected_component(active, energy):
    active = np.asarray(active, dtype=bool)
    if not active.any():
        return active

    energy = np.asarray(energy, dtype=np.float32)
    peak_flat = int(np.argmax(np.where(active, energy, -np.inf)))
    peak = np.unravel_index(peak_flat, active.shape)
    keep = np.zeros_like(active, dtype=bool)
    stack = [peak]
    keep[peak] = True

    h, w = active.shape
    while stack:
        y, x = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy = y + dy
                xx = x + dx
                if yy < 0 or yy >= h or xx < 0 or xx >= w:
                    continue
                if active[yy, xx] and not keep[yy, xx]:
                    keep[yy, xx] = True
                    stack.append((yy, xx))
    return keep


def _mask_boundary(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            interior &= padded[1 + dy:1 + dy + mask.shape[0],
                               1 + dx:1 + dx + mask.shape[1]]
    return mask & ~interior


def _convex_hull_rc(indices):
    """Return convex-hull vertices as (row, col) integer pairs."""
    pts = sorted({(int(c), int(r)) for r, c in indices})
    if len(pts) <= 2:
        return [(r, c) for c, r in pts]

    def cross(o, a, b):
        return ((a[0] - o[0]) * (b[1] - o[1])
                - (a[1] - o[1]) * (b[0] - o[0]))

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return [(r, c) for c, r in hull]


def _angle_downsample_rc(indices, max_points):
    indices = [(int(r), int(c)) for r, c in indices]
    if max_points <= 0 or len(indices) <= max_points:
        return indices
    arr = np.asarray(indices, dtype=np.float32)
    center = arr.mean(axis=0)
    angles = np.arctan2(arr[:, 0] - center[0], arr[:, 1] - center[1])
    order = np.argsort(angles)
    take = np.linspace(0, len(order) - 1, max_points)
    take = np.unique(np.round(take).astype(int))
    return [indices[int(order[i])] for i in take[:max_points]]


def _triplets_from_grid_indices(grid, indices, coord_fn):
    triplets = []
    seen = set()
    for r, c in indices:
        r = int(r)
        c = int(c)
        if (r, c) in seen:
            continue
        seen.add((r, c))
        x, y = coord_fn(r, c)
        triplets.append([
            round(float(np.clip(x, 0.0, IMG_W)), 4),
            round(float(np.clip(y, 0.0, IMG_H)), 4),
            round(float(grid[r, c]), 6),
        ])
    triplets.sort(key=lambda t: -t[2])
    return triplets


def _top_energy_indices(grid, mask, limit, exclude=None):
    if limit <= 0:
        return []
    mask = np.asarray(mask, dtype=bool)
    if exclude:
        mask = mask.copy()
        for r, c in exclude:
            if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]:
                mask[r, c] = False
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return []
    vals = grid[idx[:, 0], idx[:, 1]]
    order = np.argsort(-vals)
    return [(int(idx[i, 0]), int(idx[i, 1])) for i in order[:limit]]


def sparsify_energy_region(
    emap,
    coord_fn,
    top_k,
    thr,
    *,
    mode="region",
    rel_thr=0.35,
    pad=0,
):
    """
    Export an energy area instead of only peak samples.

    The official evaluator linearly interpolates the exported triplets and
    only fills their convex hull.  Pure top-k points near a peak therefore
    render as a tiny island.  This helper exports the active energy boundary
    (or its convex hull) plus high-energy interior points so interpolation
    forms a proper energy blob.
    """
    grid = np.asarray(emap, dtype=np.float32)
    if grid.ndim != 2 or grid.size == 0:
        return []

    max_points = int(top_k)
    if max_points <= 0:
        max_points = int(grid.size)
    max_points = max(max_points, 1)

    peak = float(np.max(grid))
    if not np.isfinite(peak) or peak <= 0.0:
        return []

    abs_thr = float(max(thr, 0.0))
    rel_thr = float(np.clip(rel_thr, 0.0, 1.0))
    active_thr = max(abs_thr, peak * rel_thr)
    active = grid >= active_thr
    if not active.any():
        active = grid >= peak

    component = _peak_connected_component(active, grid)
    if int(component.sum()) < 3:
        component = active
    if int(component.sum()) < 3:
        return []

    if pad and int(pad) > 0:
        component = _dilate_mask(component, int(pad))

    boundary = _mask_boundary(component)
    boundary_idx = [(int(r), int(c)) for r, c in np.argwhere(boundary)]
    component_idx = [(int(r), int(c)) for r, c in np.argwhere(component)]

    mode = str(mode or "region").strip().lower()
    if mode == "hull":
        outer_idx = _convex_hull_rc(boundary_idx or component_idx)
    else:
        boundary_budget = max(8, int(round(max_points * 0.65)))
        outer_idx = _angle_downsample_rc(boundary_idx or component_idx, boundary_budget)

    if len(outer_idx) > max_points:
        outer_idx = _angle_downsample_rc(outer_idx, max_points)

    selected = list(outer_idx)
    selected_set = set(selected)
    selected.extend(
        _top_energy_indices(
            grid,
            component,
            max_points - len(selected),
            exclude=selected_set,
        )
    )

    if len(selected) < 3:
        selected.extend(
            _top_energy_indices(grid, np.ones_like(component, dtype=bool), 3 - len(selected))
        )

    return _triplets_from_grid_indices(grid, selected[:max_points], coord_fn)


def export_roi_energy(emap, box, top_k, thr, export_mode="topk",
                      region_rel_thr=0.35, region_pad=0):
    if isinstance(emap, torch.Tensor):
        emap = emap.numpy()
    emap = np.asarray(emap, dtype=np.float32)
    x0, y0, x1, y1 = [float(v) for v in box]

    if str(export_mode or "topk").lower() == "topk":
        return sparsify_energy_map(emap, box, top_k, thr)

    def coord_fn(rj, ci):
        return (
            x0 + (ci + 0.5) / max(emap.shape[1], 1) * (x1 - x0),
            y0 + (rj + 0.5) / max(emap.shape[0], 1) * (y1 - y0),
        )

    triplets = sparsify_energy_region(
        emap,
        coord_fn,
        top_k,
        thr,
        mode=export_mode,
        rel_thr=region_rel_thr,
        pad=region_pad,
    )
    return triplets or sparsify_energy_map(emap, box, top_k, thr)


def sparsify_fullmap_box(
    full_energy,
    box,
    top_k,
    thr,
    fallback_to_box_center=True,
    export_mode="topk",
    region_rel_thr=0.35,
    region_pad=0,
):
    """Sparsify the full-frame energy map inside a detection box."""
    if isinstance(full_energy, torch.Tensor):
        full_energy = full_energy.detach().cpu().numpy()
    full_energy = np.asarray(full_energy, dtype=np.float32)
    if full_energy.ndim != 2 or full_energy.size == 0:
        return sparsify_energy_map(np.zeros((28, 28), dtype=np.float32), box, top_k, thr)

    x0, y0, x1, y1 = [float(v) for v in box]
    ix0 = int(np.clip(np.floor(x0), 0, IMG_W - 1))
    iy0 = int(np.clip(np.floor(y0), 0, IMG_H - 1))
    ix1 = int(np.clip(np.ceil(x1), ix0 + 1, IMG_W))
    iy1 = int(np.clip(np.ceil(y1), iy0 + 1, IMG_H))

    crop = full_energy[iy0:iy1, ix0:ix1]
    if crop.size == 0:
        return sparsify_energy_map(np.zeros((28, 28), dtype=np.float32), box, top_k, thr)

    if str(export_mode or "topk").strip().lower() != "topk":
        def coord_fn(row, col):
            return ix0 + col + 0.5, iy0 + row + 0.5

        triplets = sparsify_energy_region(
            crop,
            coord_fn,
            top_k,
            thr,
            mode=export_mode,
            rel_thr=region_rel_thr,
            pad=region_pad,
        )
        if triplets:
            return triplets

    flat = crop.reshape(-1)
    n_pts = flat.shape[0]
    top_idx = (
        np.argpartition(flat, -top_k)[-top_k:]
        if 0 < top_k < n_pts
        else np.arange(n_pts)
    )
    top_idx = top_idx[flat[top_idx] >= thr]

    triplets = []
    for idx in top_idx:
        row = int(idx // crop.shape[1])
        col = int(idx % crop.shape[1])
        triplets.append([
            round(ix0 + col + 0.5, 4),
            round(iy0 + row + 0.5, 4),
            round(float(flat[idx]), 6),
        ])

    if not triplets and fallback_to_box_center:
        peak_idx = int(flat.argmax())
        row = int(peak_idx // crop.shape[1])
        col = int(peak_idx % crop.shape[1])
        triplets.append([
            round(ix0 + col + 0.5, 4),
            round(iy0 + row + 0.5, 4),
            round(float(flat[peak_idx]), 6),
        ])

    triplets.sort(key=lambda t: -t[2])
    return triplets


# ══════════════════════════════════════════════════════════════════════════════
#  PER-SEQUENCE INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_inference_on_sequence(
    model, seq_dir, seq_name, frame_indices,
    score_thr, nms_iou_thr, max_dets_per_class,
    min_hits, iou_thr, max_age, batch_size, num_workers,
    coast_decay=COAST_DECAY,
    gpu_acoustic=False,
    acoustic_extractor=None,
    speech_prior_extractor=None,
):
    assert not model.training
    loader  = build_loader(
        seq_dir, seq_name, frame_indices, batch_size, num_workers,
        compute_acoustic=not gpu_acoustic,
    )
    tracker = InstanceTracker(iou_thr=iou_thr, max_age=max_age)
    results = []
    track_score_memory: dict = {}  
    use_amp = torch.cuda.is_available()
    n_raw = n_kept = 0
    seq_t0 = time.time()
    wav_path = wav_path_from_seq_dir(seq_dir, FRAMES_BASE, MIC_BASE)

    if gpu_acoustic and acoustic_extractor is None:
        acoustic_extractor = AcousticFeatureExtractor(
            uplam_checkpoint=UPLAM_CHECKPOINT,
            device=DEVICE,
            num_bands=N_ACOUSTIC,
        )

    with torch.no_grad():
        pbar = tqdm(loader, total=math.ceil(len(frame_indices)/batch_size),
                    desc=f"  {seq_name[:35]}", unit="batch",
                    dynamic_ncols=True, leave=True)

        for batch_fi, batch_tensors in pbar:
            if gpu_acoustic:
                rgbs_gpu = [t.to(DEVICE, non_blocking=True) for t in batch_tensors]
                acoustic_gpu = acoustic_extractor.get_frame_bands_batch(
                    [wav_path] * len(batch_fi),
                    [int(fi) for fi in batch_fi],
                    output_device=DEVICE,
                )
                images_gpu = [
                    torch.cat([rgb, acoustic], dim=0)
                    for rgb, acoustic in zip(rgbs_gpu, acoustic_gpu)
                ]
            else:
                images_gpu = [t.to(DEVICE, non_blocking=True) for t in batch_tensors]

            speech_prior_features = None
            if speech_prior_extractor is not None:
                speech_prior_features = speech_prior_extractor.get_frame_embeddings_batch(
                    [wav_path] * len(batch_fi),
                    [int(fi) for fi in batch_fi],
                    output_device=DEVICE,
                )

            with torch.amp.autocast("cuda", enabled=use_amp):
                preds_batch = model(images_gpu, None, speech_prior_features)

            for fi, det_raw in zip(batch_fi, preds_batch):
                det = {k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                       for k, v in det_raw.items()}

                n_raw  += int(det.get("scores", torch.zeros(0)).shape[0])
                det     = filter_detections(det, score_thr, nms_iou_thr,
                                            max_dets_per_class)
                n_kept += int(det.get("scores", torch.zeros(0)).shape[0])

                tracked = tracker.update(det)

                # ── Score memory: Uses 'coasting' flag to bypass decay properly ──
                for obj in tracked:
                    tid = obj["track_id"]
                    
                    if not obj.get("coasting", False):
                        track_score_memory[tid] = float(obj.get("score", 0.0))
                    else:
                        prev = track_score_memory.get(tid, 0.0)
                        decayed = prev * coast_decay
                        track_score_memory[tid] = decayed
                        obj["score"] = decayed

                    obj["hits"] = (tracker.tracks[tid]["hits"]
                                   if tid in tracker.tracks else obj.get("hits", 1))

                full_e = (det["full_energy"].numpy()
                          if "full_energy" in det
                          else np.zeros((IMG_H, IMG_W), dtype=np.float32))

                results.append(dict(
                    frame       = int(fi),
                    time_s      = int(fi) / INFERENCE_HZ,
                    objects     = tracked,
                    full_energy = full_e,
                ))

            elapsed = max(time.time() - seq_t0, 1e-6)
            pbar.set_postfix({
                "fps":    f"{len(results)/elapsed:.1f}",
                "raw":    n_raw,
                "kept":   n_kept,
                "tracks": len(tracker.tracks),
            }, refresh=True)

    elapsed = time.time() - seq_t0
    n = len(frame_indices)
    print(f"  [INFO] {n} frames in {elapsed:.1f}s  ({n/max(elapsed,1e-6):.1f} fps) | "
          f"raw={n_raw}  after_filter={n_kept}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORT GATING + JSON SERIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def collect_track_stats(results):
    stats: dict = {}
    for frame_result in results:
        for obj in frame_result["objects"]:
            tid = int(obj["track_id"])
            score = float(obj.get("score", 0.0))
            stat = stats.setdefault(tid, {
                "track_id": tid,
                "label": int(obj["label"]) - 1,
                "max_hits": 0,
                "total_frames": 0,
                "det_frames": 0,
                "score_sum": 0.0,
                "mean_det_score": 0.0,
                "max_det_score": 0.0,
            })
            stat["max_hits"] = max(stat["max_hits"], int(obj.get("hits", 1)))
            stat["total_frames"] += 1
            if not obj.get("coasting", False):
                stat["det_frames"] += 1
                stat["score_sum"] += score
                stat["max_det_score"] = max(stat["max_det_score"], score)

    for stat in stats.values():
        det_frames = int(stat["det_frames"])
        stat["mean_det_score"] = (
            float(stat["score_sum"]) / det_frames if det_frames > 0 else 0.0
        )
    return stats


def build_export_view(
    results,
    *,
    min_hits,
    track_mean_score_thr=0.0,
    track_peak_score_thr=0.0,
    track_mean_score_thr_by_class=None,
    track_peak_score_thr_by_class=None,
    max_total_dets_per_frame=0,
    max_total_dets_per_frame_by_class=None,
):
    track_mean_score_thr_by_class = parse_class_value_map(
        track_mean_score_thr_by_class,
        value_type=float,
        arg_name="track_mean_score_thr_by_class",
    )
    track_peak_score_thr_by_class = parse_class_value_map(
        track_peak_score_thr_by_class,
        value_type=float,
        arg_name="track_peak_score_thr_by_class",
    )
    max_total_dets_per_frame_by_class = parse_class_value_map(
        max_total_dets_per_frame_by_class,
        value_type=int,
        arg_name="max_total_dets_per_frame_by_class",
    )

    track_stats = collect_track_stats(results)
    selected_track_ids = set()
    for tid, stat in track_stats.items():
        label = int(stat["label"])
        mean_thr = float(track_mean_score_thr_by_class.get(label, track_mean_score_thr))
        peak_thr = float(track_peak_score_thr_by_class.get(label, track_peak_score_thr))
        if stat["max_hits"] < int(min_hits):
            continue
        if stat["mean_det_score"] < mean_thr:
            continue
        if stat["max_det_score"] < peak_thr:
            continue
        selected_track_ids.add(tid)

    exported_frames = []
    exported_track_ids = set()
    exported_obj_count = 0
    active_frames = 0

    for frame_result in results:
        objects = [
            obj for obj in frame_result["objects"]
            if int(obj["track_id"]) in selected_track_ids
        ]
        objects.sort(
            key=lambda obj: (float(obj.get("score", 0.0)), int(obj.get("hits", 1))),
            reverse=True,
        )
        if max_total_dets_per_frame_by_class:
            kept_per_class = defaultdict(int)
            class_limited = []
            for obj in objects:
                cid = int(obj["label"]) - 1
                class_cap = int(max_total_dets_per_frame_by_class.get(cid, 0))
                if class_cap > 0 and kept_per_class[cid] >= class_cap:
                    continue
                class_limited.append(obj)
                kept_per_class[cid] += 1
            objects = class_limited
        if max_total_dets_per_frame and max_total_dets_per_frame > 0:
            objects = objects[:max_total_dets_per_frame]

        if objects:
            active_frames += 1
            exported_track_ids.update(int(obj["track_id"]) for obj in objects)
        exported_obj_count += len(objects)

        exported_frames.append({
            "frame": frame_result["frame"],
            "time_s": frame_result["time_s"],
            "objects": objects,
            "full_energy": frame_result["full_energy"],
        })

    return {
        "frames": exported_frames,
        "track_stats": track_stats,
        "summary": {
            "min_hits": int(min_hits),
            "track_mean_score_thr": float(track_mean_score_thr),
            "track_peak_score_thr": float(track_peak_score_thr),
            "track_mean_score_thr_by_class": {
                str(cid): float(value)
                for cid, value in sorted(track_mean_score_thr_by_class.items())
            },
            "track_peak_score_thr_by_class": {
                str(cid): float(value)
                for cid, value in sorted(track_peak_score_thr_by_class.items())
            },
            "max_total_dets_per_frame": int(max_total_dets_per_frame),
            "max_total_dets_per_frame_by_class": {
                str(cid): int(value)
                for cid, value in sorted(max_total_dets_per_frame_by_class.items())
            },
            "num_tracks_total": len(track_stats),
            "num_tracks_after_track_gate": len(selected_track_ids),
            "num_tracks_exported": len(exported_track_ids),
            "num_objects_exported": exported_obj_count,
            "num_active_frames": active_frames,
        },
    }


def dump_inference_json(
    export_view,
    save_path,
    energy_top_k,
    energy_export_thr,
    energy_source="roi",
    energy_export_mode="topk",
    energy_region_rel_thr=0.35,
    energy_region_pad=0,
    max_json_mb=DEFAULT_MAX_JSON_MB,
):
    frames = export_view["frames"]
    energy_source = str(energy_source or "roi").strip().lower()
    if energy_source not in {"roi", "fullmap_box"}:
        raise ValueError(
            f"Unsupported energy_source={energy_source!r}; "
            "expected 'roi' or 'fullmap_box'."
        )

    annotations = []
    for frame_result in frames:
        fi = frame_result["frame"]
        for obj in frame_result["objects"]:
            if energy_source == "fullmap_box":
                triplets = sparsify_fullmap_box(
                    obj.get("full_energy", frame_result.get("full_energy")),
                    obj["box"],
                    energy_top_k,
                    energy_export_thr,
                    export_mode=energy_export_mode,
                    region_rel_thr=energy_region_rel_thr,
                    region_pad=energy_region_pad,
                )
            else:
                triplets = export_roi_energy(
                    obj["energy_map"],
                    obj["box"],
                    energy_top_k,
                    energy_export_thr,
                    export_mode=energy_export_mode,
                    region_rel_thr=energy_region_rel_thr,
                    region_pad=energy_region_pad,
                )
            annotations.append({
                "metadata_frame_index": int(fi),
                "instance_id"         : int(obj["track_id"]),
                "category_id"         : int(obj["label"]) - 1,
                "score"               : round(float(obj.get("score", 0.0)), 6),
                "distance"            : round(float(obj["dist_pred"]) * DIST_NORM, 4),
                "segmentation"        : [triplets],
            })

    def _dump_payload(payload):
        # Compact separators are valid JSON and usually save enough space to
        # stay inside the challenge evaluator's 20 MB per-file limit.
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))

    def _annotation_priority(ann):
        seg = ann.get("segmentation") or []
        n_points = sum(len(poly) for poly in seg if isinstance(poly, list))
        return (
            float(ann.get("score", 0.0)),
            int(ann.get("metadata_frame_index", 0)),
            int(ann.get("instance_id", 0)),
            n_points,
        )

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    payload = {"annotations": annotations}
    _dump_payload(payload)

    max_json_bytes = int(float(max_json_mb or 0.0) * 1024 * 1024)
    pruned = 0
    original_annotations = len(annotations)
    if max_json_bytes > 0:
        size = os.path.getsize(save_path)
        if size > max_json_bytes and annotations:
            ranked = sorted(annotations, key=_annotation_priority, reverse=True)
            lo, hi = 0, len(ranked)
            best = ranked
            # Keep the largest high-score prefix that fits. Binary search keeps
            # the protection cheap even for dense long sequences.
            while lo <= hi:
                mid = (lo + hi) // 2
                candidate = {"annotations": ranked[:mid]}
                _dump_payload(candidate)
                if os.path.getsize(save_path) <= max_json_bytes:
                    best = ranked[:mid]
                    lo = mid + 1
                else:
                    hi = mid - 1
            annotations = sorted(
                best,
                key=lambda ann: (
                    int(ann.get("metadata_frame_index", 0)),
                    int(ann.get("instance_id", 0)),
                    -float(ann.get("score", 0.0)),
                ),
            )
            pruned = original_annotations - len(annotations)
            payload = {"annotations": annotations}
            _dump_payload(payload)
            print(
                f"  [JSON] size guard: pruned {pruned}/{original_annotations} "
                f"low-score annotations to fit <= {float(max_json_mb):.2f} MiB "
                f"({os.path.getsize(save_path)/1024/1024:.2f} MiB)"
            )

    n_frames_out = len({a["metadata_frame_index"] for a in annotations})
    print(f"  [JSON] {len(annotations):5d} annotations across "
          f"{n_frames_out:4d} frames  →  {save_path}")
    return {
        "num_annotations": len(annotations),
        "num_annotations_before_size_guard": original_annotations,
        "num_annotations_pruned_by_size_guard": pruned,
        "json_size_bytes": os.path.getsize(save_path),
        "max_json_mb": float(max_json_mb or 0.0),
        "num_frames_out": n_frames_out,
        "energy_source": energy_source,
        "energy_export_mode": str(energy_export_mode or "topk").strip().lower(),
        "energy_region_rel_thr": float(energy_region_rel_thr),
        "energy_region_pad": int(energy_region_pad),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(export_view, seq_name):
    frames = export_view["frames"]
    summary = export_view["summary"]
    confirmed = [o for r in frames for o in r["objects"]]

    n_frames = len(frames)
    n_active = int(summary["num_active_frames"])
    cat_counts: dict = defaultdict(int)
    for o in confirmed:
        cat_counts[int(o["label"]) - 1] += 1

    duration_s = max((r["time_s"] for r in frames), default=0.0)
    max_total = summary["max_total_dets_per_frame"]
    max_total_str = str(max_total) if max_total > 0 else "off"
    mean_by_class = summary.get("track_mean_score_thr_by_class") or {}
    peak_by_class = summary.get("track_peak_score_thr_by_class") or {}
    cap_by_class = summary.get("max_total_dets_per_frame_by_class") or {}

    print(f"\n  +-- {seq_name}")
    print(f"  |  Frames          : {n_frames:>6d}  ({duration_s:.1f}s)")
    print(f"  |  Active frames   : {n_active:>6d}  ({100*n_active/max(n_frames,1):.1f}%)")
    print(f"  |  Confirmed objs  : {len(confirmed):>6d}")
    print(f"  |  Unique tracks   : {summary['num_tracks_exported']:>6d}")
    print(
        "  |  Track gate      : "
        f"hits>={summary['min_hits']}  "
        f"mean>={summary['track_mean_score_thr']:.2f}  "
        f"peak>={summary['track_peak_score_thr']:.2f}  "
        f"max/frame={max_total_str}"
    )
    print(
        "  |  Tracks kept     : "
        f"{summary['num_tracks_after_track_gate']:>6d}/{summary['num_tracks_total']:<6d} "
        f"(exported={summary['num_tracks_exported']})"
    )
    if mean_by_class or peak_by_class or cap_by_class:
        print(
            "  |  Class overrides : "
            f"mean[{format_class_value_map(mean_by_class)}]  "
            f"peak[{format_class_value_map(peak_by_class)}]  "
            f"cap[{format_class_value_map(cap_by_class, value_fmt='d')}]"
        )
    if cat_counts:
        print(f"  |  Category counts : "
              + "  ".join(f"C{c}:{n}" for c, n in sorted(cat_counts.items())))
    print(f"  +{'--'*25}")


def resolve_sequence_frame_indices(json_path, seq_dir, seq_name):
    try:
        with open(json_path) as jf:
            ann_data = json.load(jf)
        annotated_ids = sorted(
            {int(a["metadata_frame_index"]) for a in ann_data.get("annotations", [])}
        )
    except Exception as exc:
        print(f"  [WARN] JSON unreadable ({exc}) — using disk frames only.")
        annotated_ids = []

    disk_ids = set(scan_available_frames(seq_dir, seq_name))
    all_ids = sorted(set(annotated_ids) | disk_ids)
    return all_ids, annotated_ids, disk_ids


def run_inference_for_sequences(
    model,
    sequence_infos,
    out_dir,
    *,
    score_thr,
    nms_iou_thr,
    max_dets_per_class,
    min_hits,
    track_mean_score_thr=0.0,
    track_peak_score_thr=0.0,
    track_mean_score_thr_by_class=None,
    track_peak_score_thr_by_class=None,
    max_total_dets_per_frame=0,
    max_total_dets_per_frame_by_class=None,
    energy_top_k,
    energy_export_thr,
    energy_source="roi",
    energy_export_mode="topk",
    energy_region_rel_thr=0.35,
    energy_region_pad=0,
    max_json_mb=DEFAULT_MAX_JSON_MB,
    iou_thr,
    max_age,
    batch_size,
    num_workers,
    coast_decay=COAST_DECAY,
    gpu_acoustic=False,
    acoustic_extractor=None,
    speech_prior_extractor=None,
):
    os.makedirs(out_dir, exist_ok=True)

    wall_t0 = time.time()
    output_files = []
    skipped = []
    sequence_summaries = []
    total_annotations = 0

    for seq_idx, (json_path, seq_dir, seq_name) in enumerate(sequence_infos, 1):
        print(f"\n{'='*60}")
        print(f"[{seq_idx:3d}/{len(sequence_infos)}]  {seq_name}")

        all_ids, annotated_ids, disk_ids = resolve_sequence_frame_indices(
            json_path, seq_dir, seq_name
        )

        if not all_ids:
            print("  [WARN] No frames found — skipping.")
            skipped.append(seq_name)
            continue

        print(
            f"  Frames: {len(all_ids)}  "
            f"(annotated={len(annotated_ids)}, on-disk={len(disk_ids)})"
        )

        results = run_inference_on_sequence(
            model=model,
            seq_dir=seq_dir,
            seq_name=seq_name,
            frame_indices=all_ids,
            score_thr=score_thr,
            nms_iou_thr=nms_iou_thr,
            max_dets_per_class=max_dets_per_class,
            min_hits=min_hits,
            iou_thr=iou_thr,
            max_age=max_age,
            batch_size=batch_size,
            num_workers=num_workers,
            coast_decay=coast_decay,
            gpu_acoustic=gpu_acoustic,
            acoustic_extractor=acoustic_extractor,
            speech_prior_extractor=speech_prior_extractor,
        )

        export_view = build_export_view(
            results,
            min_hits=min_hits,
            track_mean_score_thr=track_mean_score_thr,
            track_peak_score_thr=track_peak_score_thr,
            track_mean_score_thr_by_class=track_mean_score_thr_by_class,
            track_peak_score_thr_by_class=track_peak_score_thr_by_class,
            max_total_dets_per_frame=max_total_dets_per_frame,
            max_total_dets_per_frame_by_class=max_total_dets_per_frame_by_class,
        )

        print_summary(export_view, seq_name)

        out_json = os.path.join(out_dir, f"{seq_name}_inference.json")
        json_info = dump_inference_json(
            export_view=export_view,
            save_path=out_json,
            energy_top_k=energy_top_k,
            energy_export_thr=energy_export_thr,
            energy_source=energy_source,
            energy_export_mode=energy_export_mode,
            energy_region_rel_thr=energy_region_rel_thr,
            energy_region_pad=energy_region_pad,
            max_json_mb=max_json_mb,
        )
        output_files.append(out_json)
        total_annotations += int(json_info["num_annotations"])
        sequence_summaries.append({
            "seq_name": seq_name,
            **export_view["summary"],
            **json_info,
        })

    wall_elapsed = time.time() - wall_t0
    print(f"\n{'='*60}")
    print(f"  DONE — {len(output_files)}/{len(sequence_infos)} sequences")
    if skipped:
        print(f"  Skipped  : {', '.join(skipped)}")
    print(f"  Wall time: {wall_elapsed:.1f}s  ({wall_elapsed/60:.1f} min)")
    print(f"  Output   : {os.path.abspath(out_dir)}")
    for fpath in output_files:
        print(
            f"    {os.path.basename(fpath):<50s}  "
            f"{os.path.getsize(fpath)/1024:.1f} KB"
        )
    print(f"{'='*60}\n")

    return {
        "output_files": output_files,
        "skipped": skipped,
        "wall_elapsed": wall_elapsed,
        "output_dir": os.path.abspath(out_dir),
        "energy_source": energy_source,
        "energy_export_mode": str(energy_export_mode or "topk").strip().lower(),
        "sequence_summaries": sequence_summaries,
        "num_annotations": total_annotations,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.gpu_acoustic is None:
        env_gpu_acoustic = os.environ.get("DCASE2026_GPU_ACOUSTIC", "").strip().lower()
        args.gpu_acoustic = torch.cuda.is_available() and env_gpu_acoustic in {
            "1", "true", "yes", "on"
        }

    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.exp_dir, "energy_seg_best.pth")

    args.out_dir = args.out_dir or os.path.join(args.exp_dir, "inference_outputs")
    os.makedirs(args.out_dir, exist_ok=True)

    sys.stdout = Logger(os.path.join(args.exp_dir, "inference.log"), sys.stdout)
    sys.stderr = Logger(os.path.join(args.exp_dir, "inference_error.log"), sys.stderr)

    print(f"\n[INFO] Device: {DEVICE}  [{_gpu_name}]")
    print(f"[INFO] Experiment Dir      : {args.exp_dir}")
    print(f"[INFO] Output directory    : {args.out_dir}")
    print(f"[INFO] batch_size          : {args.batch_size}")
    print(f"[INFO] num_workers         : {args.num_workers}")
    print(f"[INFO] gpu_acoustic        : {args.gpu_acoustic}")
    print(
        f"[INFO] roi_score_calib    : temp={args.roi_infer_score_temperature} | "
        f"fg_bias={args.roi_infer_fg_bias} | power={args.roi_infer_score_power}"
    )
    print(f"\n[INFO] ── Filtering pipeline ──────────────────────────────────")
    print(f"[INFO]   Stage 1  score_thr          = {args.score_thr}")
    print(f"[INFO]   Stage 2  nms_iou_thr        = {args.nms_iou_thr}")
    print(f"[INFO]   Stage 3  max_dets_per_class = {args.max_dets_per_class}")
    print(f"[INFO]   Stage 4  min_hits           = {args.min_hits}")
    print(f"[INFO]   Stage 5  track_mean_score   = {args.track_mean_score_thr}")
    print(f"[INFO]            track_peak_score   = {args.track_peak_score_thr}")
    print(f"[INFO]            mean_by_class      = {args.track_mean_score_thr_by_class or 'off'}")
    print(f"[INFO]            peak_by_class      = {args.track_peak_score_thr_by_class or 'off'}")
    print(f"[INFO]            max_total/frame    = {args.max_total_dets_per_frame or 'off'}")
    print(f"[INFO]            cap_by_class       = {args.max_total_dets_per_frame_by_class or 'off'}")
    print(f"[INFO]   Stage 6  energy_top_k       = {args.energy_top_k}")
    print(f"[INFO]            energy_export_thr  = {args.energy_export_thr}")
    print(f"[INFO]            energy_source      = {args.energy_source}")
    print(f"[INFO]            energy_export_mode = {args.energy_export_mode}")
    print(f"[INFO]            region_rel_thr     = {args.energy_region_rel_thr}")
    print(f"[INFO]            region_pad         = {args.energy_region_pad}")
    print(f"[INFO]            max_json_mb        = {args.max_json_mb}")
    print(f"[INFO] ── Tracker ─────────────────────────────────────────────")
    print(f"[INFO]   iou_thr     = {args.iou_thr}")
    print(f"[INFO]   max_age     = {args.max_age}")
    print(f"[INFO]   coast_decay = {args.coast_decay}  "
          f"(coasting-frame score multiplier per frame)")
    print(f"[INFO] ────────────────────────────────────────────────────────\n")

    model      = load_model(
        args.checkpoint,
        args.num_classes,
        roi_infer_score_temperature=args.roi_infer_score_temperature,
        roi_infer_fg_bias=args.roi_infer_fg_bias,
        roi_infer_score_power=args.roi_infer_score_power,
    )
    speech_prior_extractor = None
    if getattr(model, "use_speech_prior", False):
        speech_prior_extractor = SpeechPriorExtractor(
            bundle_name=getattr(model, "speech_prior_bundle", "WAVLM_BASE_PLUS"),
            device=DEVICE,
            context_sec=getattr(model, "speech_prior_context_sec", 1.0),
            cache_size=int(os.environ.get("DCASE2026_SPEECH_PRIOR_CACHE_SIZE", "512")),
            use_amp=torch.cuda.is_available(),
        )
    gpu_acoustic_extractor = None
    if args.gpu_acoustic:
        gpu_acoustic_extractor = AcousticFeatureExtractor(
            uplam_checkpoint=UPLAM_CHECKPOINT,
            device=DEVICE,
            num_bands=N_ACOUSTIC,
        )
    test_infos = get_sequence_infos(args.split, LABELS_BASE, FRAMES_BASE)

    if not test_infos:
        print(f"[ERROR] No sequences found — split='{args.split}'")
        sys.exit(1)

    if args.seqs.strip():
        wanted = [item.strip() for item in args.seqs.split(",") if item.strip()]
        by_name = {seq_name: info for info in test_infos for seq_name in [info[2]]}
        missing = [name for name in wanted if name not in by_name]
        if missing:
            print(f"[ERROR] Requested sequence(s) not found: {missing}")
            sys.exit(1)
        test_infos = [by_name[name] for name in wanted]
        print(f"[INFO] Explicit sequence list: {', '.join(wanted)}")
    elif args.num_seqs is not None:
        if args.num_seqs >= len(test_infos):
            print(f"[INFO] Requested num_seqs ({args.num_seqs}) >= available "
                  f"({len(test_infos)}). Using all.")
        else:
            print(f"[INFO] Randomly sampling {args.num_seqs} sequences "
                  f"out of {len(test_infos)}")
            test_infos = random.sample(test_infos, args.num_seqs)

    print(f"[INFO] {len(test_infos)} sequences to process.\n")

    run_inference_for_sequences(
        model=model,
        sequence_infos=test_infos,
        out_dir=args.out_dir,
        score_thr=args.score_thr,
        nms_iou_thr=args.nms_iou_thr,
        max_dets_per_class=args.max_dets_per_class,
        min_hits=args.min_hits,
        track_mean_score_thr=args.track_mean_score_thr,
        track_peak_score_thr=args.track_peak_score_thr,
        track_mean_score_thr_by_class=args.track_mean_score_thr_by_class,
        track_peak_score_thr_by_class=args.track_peak_score_thr_by_class,
        max_total_dets_per_frame=args.max_total_dets_per_frame,
        max_total_dets_per_frame_by_class=args.max_total_dets_per_frame_by_class,
        energy_top_k=args.energy_top_k,
        energy_export_thr=args.energy_export_thr,
        energy_source=args.energy_source,
        energy_export_mode=args.energy_export_mode,
        energy_region_rel_thr=args.energy_region_rel_thr,
        energy_region_pad=args.energy_region_pad,
        max_json_mb=args.max_json_mb,
        iou_thr=args.iou_thr,
        max_age=args.max_age,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        coast_decay=args.coast_decay,
        gpu_acoustic=args.gpu_acoustic,
        acoustic_extractor=gpu_acoustic_extractor,
        speech_prior_extractor=speech_prior_extractor,
    )


if __name__ == "__main__":
    main()
