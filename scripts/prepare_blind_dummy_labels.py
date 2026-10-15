#!/usr/bin/env python3
"""Create dummy JSON files only for blind sequence enumeration.

The 2026 inference script discovers sequences from labels_base/*.json.  Blind
evaluation has no labels, so this creates empty JSON placeholders with matching
sequence names. These files are never used for metric evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metadata-dir", required=True)
    p.add_argument("--out-labels-base", required=True)
    p.add_argument("--split-name", default="dev-test-blind")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metadata_dir = Path(args.metadata_dir)
    out_split = Path(args.out_labels_base) / args.split_name
    out_split.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(metadata_dir.glob("*.csv"))
    for path in csv_paths:
        out_path = out_split / f"{path.stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"annotations": []}, f, separators=(",", ":"))

    print(f"[dummy-labels] wrote {len(csv_paths)} JSON files to {out_split}")


if __name__ == "__main__":
    main()

