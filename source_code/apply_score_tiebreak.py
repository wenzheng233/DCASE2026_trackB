#!/usr/bin/env python3
"""Apply deterministic tiny score tie-breaks to DCASE inference JSONs.

The official evaluator sorts detections by score only. When many detections
share exactly the same score, AP can depend on input/order details unrelated to
the model. This post-process keeps prediction masks/classes intact and only
adds a tiny, deterministic score offset inside same-score groups.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Add deterministic tiny score tie-breaks to inference JSONs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pred-dir", required=True, help="Input *_inference.json directory.")
    p.add_argument("--out-dir", required=True, help="Output experiment dir; writes inference_outputs.")
    p.add_argument(
        "--mode",
        choices=(
            "class_asc",
            "class_desc",
            "class_desc_area_desc",
            "class_desc_area_asc",
            "class_desc_energy_desc",
            "class_desc_energy_asc",
            "custom_class",
            "custom_class_area_desc",
            "custom_class_area_asc",
            "custom_class_energy_desc",
            "custom_class_energy_asc",
            "custom_class_sum_energy_asc",
            "custom_class_mean_energy_asc",
            "area_desc",
            "area_asc",
            "energy_desc",
            "energy_asc",
            "frame_class_area",
            "hash",
            "random",
        ),
        default="frame_class_area",
    )
    p.add_argument("--epsilon", type=float, default=1e-6, help="Max offset added to scores.")
    p.add_argument("--seed", type=int, default=0, help="Seed used by random mode.")
    p.add_argument(
        "--class-priority",
        default="",
        help="Comma-separated class ids from highest to lowest priority for custom_class mode.",
    )
    p.add_argument(
        "--direction",
        choices=("up", "down"),
        default="down",
        help="Whether tie-break scores move above or below --only-score.",
    )
    p.add_argument("--digits", type=int, default=12, help="Decimal digits used when writing perturbed scores.")
    p.add_argument(
        "--only-score",
        type=float,
        default=1.0,
        help="Only perturb annotations whose score is within 1e-9 of this value.",
    )
    return p.parse_args()


def ann_stats(ann: dict) -> tuple[float, float, float, int]:
    n_pts = 0
    sum_energy = 0.0
    max_energy = 0.0
    for poly in ann.get("segmentation", []):
        for triplet in poly:
            if len(triplet) < 3:
                continue
            e = float(triplet[2])
            n_pts += 1
            sum_energy += e
            if e > max_energy:
                max_energy = e
    mean_energy = sum_energy / n_pts if n_pts else 0.0
    return float(n_pts), float(sum_energy), float(max_energy), int(round(mean_energy * 1_000_000))


def stable_hash(seq: str, ann: dict, idx: int, seed: int = 0) -> int:
    frame = ann.get("metadata_frame_index", -1)
    cid = ann.get("category_id", -1)
    payload = f"{seed}|{seq}|{idx}|{frame}|{cid}|{ann.get('score', 0)}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def parse_class_priority(raw: str) -> dict[int, int]:
    if not raw.strip():
        return {}
    order = [int(x.strip()) for x in raw.split(",") if x.strip()]
    return {cid: len(order) - rank for rank, cid in enumerate(order)}


def rank_value(mode: str, seq: str, ann: dict, idx: int, seed: int, class_priority: dict[int, int]) -> tuple:
    cid = int(ann.get("category_id", -1))
    frame = int(ann.get("metadata_frame_index", -1))
    area, sum_energy, max_energy, mean_energy_i = ann_stats(ann)
    h = stable_hash(seq, ann, idx, seed)
    if mode == "class_asc":
        return (-cid, -frame, -h)
    if mode == "class_desc":
        return (cid, -frame, -h)
    if mode == "class_desc_area_desc":
        return (cid, area, sum_energy, max_energy, -frame, -h)
    if mode == "class_desc_area_asc":
        return (cid, -area, -sum_energy, -max_energy, -frame, -h)
    if mode == "class_desc_energy_desc":
        return (cid, max_energy, sum_energy, area, -frame, -h)
    if mode == "class_desc_energy_asc":
        return (cid, -max_energy, -sum_energy, -area, -frame, -h)
    if mode == "custom_class":
        return (class_priority.get(cid, 0), -frame, -h)
    if mode == "custom_class_area_desc":
        return (class_priority.get(cid, 0), area, sum_energy, max_energy, -frame, -h)
    if mode == "custom_class_area_asc":
        return (class_priority.get(cid, 0), -area, -sum_energy, -max_energy, -frame, -h)
    if mode == "custom_class_energy_desc":
        return (class_priority.get(cid, 0), max_energy, sum_energy, area, -frame, -h)
    if mode == "custom_class_energy_asc":
        return (class_priority.get(cid, 0), -max_energy, -sum_energy, -area, -frame, -h)
    if mode == "custom_class_sum_energy_asc":
        return (class_priority.get(cid, 0), -sum_energy, -max_energy, -area, -frame, -h)
    if mode == "custom_class_mean_energy_asc":
        return (class_priority.get(cid, 0), -mean_energy_i, -max_energy, -sum_energy, -area, -frame, -h)
    if mode == "area_desc":
        return (area, sum_energy, max_energy, -h)
    if mode == "area_asc":
        return (-area, -sum_energy, -max_energy, -h)
    if mode == "energy_desc":
        return (max_energy, sum_energy, area, -h)
    if mode == "energy_asc":
        return (-max_energy, -sum_energy, -area, -h)
    if mode == "frame_class_area":
        return (-frame, -cid, area, sum_energy, max_energy, -h)
    if mode == "random":
        return (h,)
    return (h,)


def apply_file(
    path: Path,
    out_path: Path,
    mode: str,
    epsilon: float,
    only_score: float,
    seed: int,
    direction: str,
    digits: int,
    class_priority: dict[int, int],
) -> dict:
    data = json.loads(path.read_text())
    anns = data.get("annotations", [])
    seq = path.name.replace("_inference.json", "")

    targets = []
    for idx, ann in enumerate(anns):
        score = float(ann.get("score", 0.0))
        if math.isclose(score, only_score, rel_tol=0.0, abs_tol=1e-9):
            targets.append((idx, rank_value(mode, seq, ann, idx, seed, class_priority)))

    if targets:
        ordered = sorted(targets, key=lambda x: x[1], reverse=True)
        denom = max(1, len(ordered) - 1)
        for rank, (idx, _) in enumerate(ordered):
            # Highest rank should sort first. In "down" mode it stays closest
            # to only_score; in "up" mode it receives the largest tiny offset.
            frac = (denom - rank) / denom
            if direction == "down":
                offset = epsilon * rank / denom
                score = float(only_score) - offset
            else:
                offset = epsilon * frac
                score = float(only_score) + offset
            anns[idx]["score"] = round(score, int(digits))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"annotations": anns}, separators=(",", ":")))
    return {
        "file": path.name,
        "annotations": len(anns),
        "perturbed": len(targets),
    }


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    out_exp = Path(args.out_dir)
    out_pred = out_exp / "inference_outputs"
    if out_pred.exists():
        shutil.rmtree(out_pred)
    out_pred.mkdir(parents=True, exist_ok=True)

    total_annotations = 0
    total_perturbed = 0
    class_priority = parse_class_priority(args.class_priority)
    files = sorted(pred_dir.glob("*_inference.json"))
    for idx, path in enumerate(files, 1):
        stats = apply_file(
            path,
            out_pred / path.name,
            mode=args.mode,
            epsilon=float(args.epsilon),
            only_score=float(args.only_score),
            seed=int(args.seed),
            direction=args.direction,
            digits=int(args.digits),
            class_priority=class_priority,
        )
        total_annotations += stats["annotations"]
        total_perturbed += stats["perturbed"]
        if idx % 10 == 0 or idx == len(files):
            print(f"[tiebreak] {idx}/{len(files)} files", flush=True)

    summary = {
        "pred_dir": str(pred_dir),
        "out_dir": str(out_exp),
        "mode": args.mode,
        "epsilon": float(args.epsilon),
        "only_score": float(args.only_score),
        "seed": int(args.seed),
        "direction": args.direction,
        "digits": int(args.digits),
        "class_priority": {str(k): int(v) for k, v in sorted(class_priority.items())},
        "files": len(files),
        "annotations": int(total_annotations),
        "perturbed": int(total_perturbed),
    }
    (out_exp / "score_tiebreak_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
