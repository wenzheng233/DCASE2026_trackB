#!/usr/bin/env python3
"""Apply DCASE2023 event-count prior to 2026 inference JSONs.

This script keeps the 2026 proposals / masks / DOA fields intact, and uses
the 2023 model only as a count prior:

  * per_class mode: keep at most N detections for each (frame, class)
    where N is the 2023 teacher count for that same frame and class
  * total mode: keep at most N detections per frame, where N is the total
    number of 2023 teacher events in that frame

Detections are sorted by score before trimming.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


N_CLS = 13
CLASS_NAME_TO_ID = {
    "Female speech": 0,
    "Male speech": 1,
    "Clapping": 2,
    "Telephone": 3,
    "Laughter": 4,
    "Domestic sounds": 5,
    "Walk/footsteps": 6,
    "Door open/close": 7,
    "Music": 8,
    "Musical instr.": 9,
    "Water tap/shower": 10,
    "Bell": 11,
    "Knock": 12,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply DCASE2023 count prior to 2026 predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pred-dir", required=True, help="Input 2026 *_inference.json directory.")
    p.add_argument("--teacher-csv-dir", required=True, help="DCASE2023 CSV prediction directory.")
    p.add_argument("--out-dir", required=True, help="Output experiment dir; writes inference_outputs inside it.")
    p.add_argument(
        "--count-mode",
        choices=("per_class", "total"),
        default="per_class",
        help="Whether to cap by per-frame per-class count or per-frame total count.",
    )
    p.add_argument(
        "--keep-order",
        action="store_true",
        help="Keep the original annotation order after trimming instead of sorting by score.",
    )
    p.add_argument(
        "--score-thr-by-class",
        default="",
        help="Optional per-class score threshold map, e.g. 'Music:0.3,Male speech:0.5'.",
    )
    p.add_argument(
        "--default-score-thr",
        type=float,
        default=0.0,
        help="Fallback score threshold for classes not listed in --score-thr-by-class.",
    )
    p.add_argument(
        "--uncapped-classes",
        default="",
        help=(
            "Optional comma-separated class list that bypasses the 2023 count cap "
            "in per_class mode, e.g. 'Bell,Knock,Water tap/shower' or '10,11,12'."
        ),
    )
    return p.parse_args()


def load_teacher_counts(csv_dir: str) -> dict[str, dict[str, dict[int, int] | dict[tuple[int, int], int]]]:
    out: dict[str, dict[str, dict[int, int] | dict[tuple[int, int], int]]] = {}
    for path in sorted(Path(csv_dir).glob("*.csv")):
        seq = path.stem
        frame_total: dict[int, int] = defaultdict(int)
        frame_class: dict[tuple[int, int], int] = defaultdict(int)
        with open(path, newline="") as f:
            for row in csv.reader(f):
                if len(row) < 6:
                    continue
                try:
                    frame = int(row[0])
                    cid = int(row[1])
                except ValueError:
                    continue
                if not (0 <= cid < N_CLS):
                    continue
                frame_total[frame] += 1
                frame_class[(frame, cid)] += 1
        out[seq] = {
            "frame_total": dict(frame_total),
            "frame_class": dict(frame_class),
        }
    return out


def parse_class_float_map(raw: str) -> dict[int, float]:
    if not raw:
        return {}
    raw = raw.strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError
    except Exception:
        parsed = {}
        for chunk in raw.split(","):
            item = chunk.strip()
            if not item:
                continue
            if ":" in item:
                key, value = item.split(":", 1)
            elif "=" in item:
                key, value = item.split("=", 1)
            else:
                raise ValueError(
                    f"Invalid score threshold entry '{item}'. Use key:value pairs or JSON."
                )
            parsed[key.strip()] = value.strip()

    out: dict[int, float] = {}
    for key, value in parsed.items():
        if key not in CLASS_NAME_TO_ID:
            try:
                cid = int(key)
            except ValueError as exc:
                raise ValueError(f"Unknown class key '{key}'.") from exc
        else:
            cid = CLASS_NAME_TO_ID[key]
        out[int(cid)] = float(value)
    return out


def parse_class_id_list(raw: str) -> set[int]:
    if not raw:
        return set()
    out: set[int] = set()
    for chunk in raw.split(","):
        key = chunk.strip()
        if not key:
            continue
        if key in CLASS_NAME_TO_ID:
            cid = CLASS_NAME_TO_ID[key]
        else:
            try:
                cid = int(key)
            except ValueError as exc:
                raise ValueError(f"Unknown class key '{key}'.") from exc
        if not (0 <= cid < N_CLS):
            raise ValueError(f"Class id out of range: {cid}")
        out.add(cid)
    return out


def ann_score(ann: dict) -> float:
    try:
        return float(ann.get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def apply_count_prior_to_file(
    pred_path: Path,
    teacher_counts: dict[str, dict[str, dict[int, int] | dict[tuple[int, int], int]]],
    out_path: Path,
    *,
    count_mode: str,
    keep_order: bool,
    score_thr_by_class: dict[int, float],
    default_score_thr: float,
    uncapped_classes: set[int],
) -> Counter:
    seq = pred_path.name.replace("_inference.json", "")
    with open(pred_path) as f:
        data = json.load(f)

    teacher_seq = teacher_counts.get(seq, {})
    frame_total = teacher_seq.get("frame_total", {})
    frame_class = teacher_seq.get("frame_class", {})

    by_frame: dict[int, list[dict]] = defaultdict(list)
    stats = Counter()

    for ann in data.get("annotations", []):
        stats["input_annotations"] += 1
        cid = int(ann.get("category_id", -1))
        score = ann_score(ann)
        thr = float(score_thr_by_class.get(cid, default_score_thr))
        if score < thr:
            stats["dropped_score"] += 1
            continue
        frame = int(ann.get("metadata_frame_index", -1))
        by_frame[frame].append(ann)

    new_annotations = []
    for frame in sorted(by_frame):
        anns = by_frame[frame]
        if not anns:
            continue

        if count_mode == "total":
            cap = int(frame_total.get(frame, 0))
            if cap <= 0:
                stats["dropped_total_zero"] += len(anns)
                continue
            if keep_order:
                kept = anns[:cap]
            else:
                kept = sorted(anns, key=ann_score, reverse=True)[:cap]
            stats["kept_total"] += len(kept)
            stats["dropped_total"] += max(0, len(anns) - len(kept))
            new_annotations.extend(kept)
            continue

        # per_class
        by_class: dict[int, list[dict]] = defaultdict(list)
        for ann in anns:
            cid = int(ann.get("category_id", -1))
            if 0 <= cid < N_CLS:
                by_class[cid].append(ann)
            else:
                stats["invalid_category"] += 1

        kept_frame = []
        for cid in sorted(by_class):
            class_anns = by_class[cid]
            if cid in uncapped_classes:
                kept = class_anns if keep_order else sorted(class_anns, key=ann_score, reverse=True)
                kept_frame.extend(kept)
                stats["kept_class_uncapped"] += len(kept)
                continue
            cap = int(frame_class.get((frame, cid), 0))
            if cap <= 0:
                stats["dropped_class_zero"] += len(class_anns)
                continue
            if keep_order:
                kept = class_anns[:cap]
            else:
                kept = sorted(class_anns, key=ann_score, reverse=True)[:cap]
            kept_frame.extend(kept)
            stats["kept_class"] += len(kept)
            stats["dropped_class"] += max(0, len(class_anns) - len(kept))

        # Keep a deterministic frame order in the output.
        if not keep_order:
            kept_frame.sort(key=ann_score, reverse=True)
        new_annotations.extend(kept_frame)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"annotations": new_annotations}, f, separators=(",", ":"))

    stats["output_annotations"] += len(new_annotations)
    return stats


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    out_exp = Path(args.out_dir)
    out_pred_dir = out_exp / "inference_outputs"

    if out_pred_dir.exists():
        shutil.rmtree(out_pred_dir)
    out_pred_dir.mkdir(parents=True, exist_ok=True)

    teacher_counts = load_teacher_counts(args.teacher_csv_dir)
    score_thr_by_class = parse_class_float_map(args.score_thr_by_class)
    uncapped_classes = parse_class_id_list(args.uncapped_classes)
    total = Counter()
    pred_files = sorted(pred_dir.glob("*_inference.json"))
    for idx, pred_path in enumerate(pred_files, 1):
        out_path = out_pred_dir / pred_path.name
        stats = apply_count_prior_to_file(
            pred_path,
            teacher_counts,
            out_path,
            count_mode=args.count_mode,
            keep_order=args.keep_order,
            score_thr_by_class=score_thr_by_class,
            default_score_thr=float(args.default_score_thr),
            uncapped_classes=uncapped_classes,
        )
        total.update(stats)
        if idx % 10 == 0 or idx == len(pred_files):
            print(f"[count-prior] {idx}/{len(pred_files)} files", flush=True)

    summary = {
        "pred_dir": str(pred_dir),
        "teacher_csv_dir": str(args.teacher_csv_dir),
        "out_dir": str(out_exp),
        "count_mode": args.count_mode,
        "keep_order": bool(args.keep_order),
        "score_thr_by_class": {
            str(cid): float(v) for cid, v in sorted(score_thr_by_class.items())
        },
        "default_score_thr": float(args.default_score_thr),
        "uncapped_classes": sorted(int(cid) for cid in uncapped_classes),
        "input_annotations": int(total["input_annotations"]),
        "output_annotations": int(total["output_annotations"]),
        "dropped_score": int(total["dropped_score"]),
        "dropped_class": int(total["dropped_class"]),
        "dropped_class_zero": int(total["dropped_class_zero"]),
        "dropped_total": int(total["dropped_total"]),
        "dropped_total_zero": int(total["dropped_total_zero"]),
        "kept_class": int(total["kept_class"]),
        "kept_class_uncapped": int(total["kept_class_uncapped"]),
        "kept_total": int(total["kept_total"]),
        "invalid_category": int(total["invalid_category"]),
    }
    with open(out_exp / "count_prior_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
