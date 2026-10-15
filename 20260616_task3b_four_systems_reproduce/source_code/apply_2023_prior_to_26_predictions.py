#!/usr/bin/env python3
"""Apply DCASE2023 AV-SELD predictions as an inference-time prior for 2026 JSONs.

The 2026 model keeps ownership of the proposal mask / energy field.  The 2023
model is used only as a same-frame, nearest-DOA class prior, with an optional
translation of the 2026 mask peak to the 2023 DOA.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

IMG_W = 360.0
IMG_H = 180.0
N_CLS = 13


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hybridize 26 mask proposals with 23 class/DOA prior.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pred-dir", required=True, help="Input 2026 *_inference.json directory.")
    p.add_argument("--teacher-csv-dir", required=True, help="DCASE2023 CSV prediction directory.")
    p.add_argument("--out-dir", required=True, help="Output experiment dir; writes inference_outputs inside it.")
    p.add_argument("--match-thr-deg", type=float, default=20.0)
    p.add_argument("--shift-to-teacher-doa", action="store_true")
    p.add_argument(
        "--coord-convention",
        choices=("stairs26", "legacy"),
        default="stairs26",
        help=(
            "Coordinate convention for 2026 JSON mask points. STAIRS26 uses "
            "x=180-azimuth, y=90-elevation; legacy keeps the older "
            "x=azimuth+180 conversion used by early diagnostics."
        ),
    )
    p.add_argument(
        "--score-mode",
        choices=("keep", "matched_one", "matched_max095"),
        default="keep",
    )
    p.add_argument(
        "--unmatched",
        choices=("keep", "drop"),
        default="keep",
        help="What to do when no 2023 prediction is close enough.",
    )
    p.add_argument(
        "--keep-teacher-metadata",
        action="store_true",
        help="Keep teacher match debug fields in each annotation. Disabled by default to preserve the 20 MB submission limit.",
    )
    return p.parse_args()


def cart_to_azel(x: float, y: float, z: float) -> tuple[float, float]:
    norm = math.sqrt(x * x + y * y + z * z) + 1e-10
    x, y, z = x / norm, y / norm, z / norm
    az = math.degrees(math.atan2(y, x))
    el = math.degrees(math.atan2(z, math.sqrt(x * x + y * y)))
    return az, el


def wrap_azimuth(deg: float) -> float:
    return ((float(deg) + 180.0) % 360.0) - 180.0


def azel_to_px(az: float, el: float, coord_convention: str) -> tuple[float, float]:
    if coord_convention == "stairs26":
        x = 180.0 - wrap_azimuth(az)
        y = 90.0 - float(el)
    else:
        x = ((az + 180.0) / 360.0) * IMG_W
        y = ((90.0 - el) / 180.0) * (IMG_H - 1.0)
    return x % IMG_W, min(max(y, 0.0), IMG_H - 1.0)


def px_to_azel(x: float, y: float, coord_convention: str) -> tuple[float, float]:
    if coord_convention == "stairs26":
        az = wrap_azimuth(180.0 - float(x))
        el = 90.0 - float(y)
        return az, el
    az = (x / IMG_W) * 360.0 - 180.0
    el = 90.0 - (y / (IMG_H - 1.0)) * 180.0
    return az, el


def angular_distance_from_azel(a: tuple[float, float], b: tuple[float, float]) -> float:
    az1, el1 = map(math.radians, a)
    az2, el2 = map(math.radians, b)
    dot = (
        math.sin(el1) * math.sin(el2)
        + math.cos(el1) * math.cos(el2) * math.cos(abs(az1 - az2))
    )
    return math.degrees(math.acos(min(1.0, max(-1.0, dot))))


def load_teacher_csvs(csv_dir: str, coord_convention: str) -> dict[str, dict[int, list[dict]]]:
    out: dict[str, dict[int, list[dict]]] = {}
    for path in sorted(Path(csv_dir).glob("*.csv")):
        seq = path.stem
        by_frame: dict[int, list[dict]] = defaultdict(list)
        with open(path, newline="") as f:
            for row in csv.reader(f):
                if len(row) < 6:
                    continue
                frame = int(row[0])
                cls = int(row[1])
                if not (0 <= cls < N_CLS):
                    continue
                x, y, z = float(row[3]), float(row[4]), float(row[5])
                az, el = cart_to_azel(x, y, z)
                px, py = azel_to_px(az, el, coord_convention)
                by_frame[frame].append({
                    "cls": cls,
                    "az": az,
                    "el": el,
                    "px": px,
                    "py": py,
                })
        out[seq] = dict(by_frame)
    return out


def annotation_peak(ann: dict) -> tuple[float, float, float]:
    best = (0.0, 0.0, -float("inf"))
    for sub in ann.get("segmentation", []):
        for triplet in sub:
            if len(triplet) < 3:
                continue
            x, y, e = float(triplet[0]), float(triplet[1]), float(triplet[2])
            if e > best[2]:
                best = (x, y, e)
    if best[2] == -float("inf"):
        return 0.0, 0.0, 0.0
    return best


def circular_dx(src_x: float, dst_x: float) -> float:
    dx = (dst_x - src_x + IMG_W / 2.0) % IMG_W - IMG_W / 2.0
    return dx


def shift_segmentation(ann: dict, dst_x: float, dst_y: float) -> None:
    src_x, src_y, _ = annotation_peak(ann)
    dx = circular_dx(src_x, dst_x)
    dy = dst_y - src_y
    shifted = []
    for sub in ann.get("segmentation", []):
        new_sub = []
        for triplet in sub:
            if len(triplet) < 3:
                continue
            x = (float(triplet[0]) + dx) % IMG_W
            y = min(max(float(triplet[1]) + dy, 0.0), IMG_H - 1.0)
            e = float(triplet[2])
            new_sub.append([round(x, 4), round(y, 4), e])
        shifted.append(new_sub)
    ann["segmentation"] = shifted


def find_teacher_match(
    ann: dict,
    teacher_frame: list[dict],
    thr: float,
    coord_convention: str,
) -> tuple[dict | None, float | None]:
    if not teacher_frame:
        return None, None
    px, py, _ = annotation_peak(ann)
    pred_azel = px_to_azel(px, py, coord_convention)
    best_teacher, best_dist = None, float("inf")
    for teacher in teacher_frame:
        dist = angular_distance_from_azel(pred_azel, (teacher["az"], teacher["el"]))
        if dist < best_dist:
            best_teacher, best_dist = teacher, dist
    if best_teacher is None or best_dist > thr:
        return None, best_dist
    return best_teacher, best_dist


def apply_prior_to_file(
    pred_path: Path,
    teacher: dict[str, dict[int, list[dict]]],
    out_path: Path,
    *,
    match_thr_deg: float,
    shift_to_teacher_doa: bool,
    score_mode: str,
    unmatched: str,
    coord_convention: str,
    keep_teacher_metadata: bool,
) -> Counter:
    seq = pred_path.name.replace("_inference.json", "")
    with open(pred_path) as f:
        data = json.load(f)

    teacher_seq = teacher.get(seq, {})
    new_annotations = []
    stats = Counter()
    dists = []

    for ann in data.get("annotations", []):
        stats["input_annotations"] += 1
        frame = int(ann.get("metadata_frame_index", -1))
        match, dist = find_teacher_match(
            ann,
            teacher_seq.get(frame, []),
            match_thr_deg,
            coord_convention,
        )
        if dist is not None:
            dists.append(dist)

        if match is None:
            stats["unmatched"] += 1
            if unmatched == "drop":
                continue
            new_annotations.append(ann)
            continue

        stats["matched"] += 1
        old_cls = int(ann.get("category_id", -1))
        new_cls = int(match["cls"])
        if old_cls != new_cls:
            stats["class_changed"] += 1
        ann["category_id"] = new_cls
        if keep_teacher_metadata:
            ann["teacher23_match_dist_deg"] = round(float(dist), 4)
            ann["teacher23_class_id"] = new_cls

        if shift_to_teacher_doa:
            shift_segmentation(ann, float(match["px"]), float(match["py"]))
            stats["shifted"] += 1

        if score_mode == "matched_one":
            ann["score"] = 1.0
        elif score_mode == "matched_max095":
            ann["score"] = max(float(ann.get("score", 0.0)), 0.95)

        new_annotations.append(ann)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"annotations": new_annotations}, f, separators=(",", ":"))

    stats["output_annotations"] += len(new_annotations)
    if dists:
        stats["sum_match_dist_deg"] += float(np.sum(dists))
        stats["num_dist_records"] += len(dists)
    return stats


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    out_exp = Path(args.out_dir)
    out_pred_dir = out_exp / "inference_outputs"

    if out_pred_dir.exists():
        shutil.rmtree(out_pred_dir)
    out_pred_dir.mkdir(parents=True, exist_ok=True)

    teacher = load_teacher_csvs(args.teacher_csv_dir, args.coord_convention)
    total = Counter()
    pred_files = sorted(pred_dir.glob("*_inference.json"))
    for idx, pred_path in enumerate(pred_files, 1):
        out_path = out_pred_dir / pred_path.name
        stats = apply_prior_to_file(
            pred_path,
            teacher,
            out_path,
            match_thr_deg=args.match_thr_deg,
            shift_to_teacher_doa=args.shift_to_teacher_doa,
            score_mode=args.score_mode,
            unmatched=args.unmatched,
            coord_convention=args.coord_convention,
            keep_teacher_metadata=bool(args.keep_teacher_metadata),
        )
        total.update(stats)
        if idx % 10 == 0 or idx == len(pred_files):
            print(f"[hybrid] {idx}/{len(pred_files)} files", flush=True)

    mean_dist = (
        total["sum_match_dist_deg"] / total["num_dist_records"]
        if total["num_dist_records"]
        else float("nan")
    )
    summary = {
        "pred_dir": str(pred_dir),
        "teacher_csv_dir": str(args.teacher_csv_dir),
        "out_dir": str(out_exp),
        "match_thr_deg": args.match_thr_deg,
        "shift_to_teacher_doa": bool(args.shift_to_teacher_doa),
        "score_mode": args.score_mode,
        "unmatched": args.unmatched,
        "coord_convention": args.coord_convention,
        "keep_teacher_metadata": bool(args.keep_teacher_metadata),
        "input_annotations": int(total["input_annotations"]),
        "output_annotations": int(total["output_annotations"]),
        "matched": int(total["matched"]),
        "unmatched_count": int(total["unmatched"]),
        "class_changed": int(total["class_changed"]),
        "shifted": int(total["shifted"]),
        "mean_nearest_teacher_dist_deg": mean_dist,
    }
    with open(out_exp / "hybrid_2023_prior_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
