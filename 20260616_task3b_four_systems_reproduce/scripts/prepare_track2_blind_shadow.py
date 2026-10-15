#!/usr/bin/env python3
"""Prepare a DCASE2026-like layout for DCASE2023 track2 blind inference.

The 2026 inference code expects:

  frames_dev/<split>/3/<sequence>.mp4/*.jpg
  foa_dev/<split>/<sequence>.wav
  metadata_dev/<split>/<sequence>.csv

The real blind data is named mix001...mix079.  This script creates the
fold9_room0_mixXXX names used by the current v16 2026 pipeline and links them
back to the track2 blind frames/audio.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--track2-root",
        default="/data/xinweiwu/DCASE_2023data",
        help="Root containing foa_eval/mic_eval/video_eval for DCASE2023 track2.",
    )
    p.add_argument(
        "--frames-dir",
        default="/data/yishuoyang/dataset/decase_challenge/frames_eval_track2_blind",
        help="Extracted blind frame directories, one <sequence>.mp4 directory per mix.",
    )
    p.add_argument(
        "--shadow-root",
        default="/tmp/avseld_dcase2023_track2_blind_shadow",
        help="Output root with 2026-like frames_dev/foa_dev/metadata_dev layout.",
    )
    p.add_argument("--split-name", default="dev-test-blind")
    p.add_argument("--camera", default="3")
    p.add_argument("--seq-prefix", default="fold9_room0_")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def safe_link(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if overwrite:
            if dst.is_dir() and not dst.is_symlink():
                # Existing frame dirs may be the source itself. Leave them alone.
                if dst.resolve() == src.resolve():
                    return
                raise IsADirectoryError(f"Refusing to remove directory: {dst}")
            dst.unlink()
        else:
            return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve())


def mix_id_from_name(path: Path) -> str | None:
    match = re.search(r"(mix\d{3})", path.stem)
    return match.group(1) if match else None


def main() -> None:
    args = parse_args()
    track2_root = Path(args.track2_root)
    frames_dir = Path(args.frames_dir)
    shadow = Path(args.shadow_root)
    split = args.split_name

    frame_out = shadow / "frames_dev" / split / args.camera
    audio_out = shadow / "foa_dev" / split
    meta_out = shadow / "metadata_dev" / split
    frame_out.mkdir(parents=True, exist_ok=True)
    audio_out.mkdir(parents=True, exist_ok=True)
    meta_out.mkdir(parents=True, exist_ok=True)

    frame_dirs = sorted(p for p in frames_dir.iterdir() if p.is_dir())
    made_frames = made_audio = made_meta = missing_audio = 0
    for frame_dir in frame_dirs:
        mix_id = mix_id_from_name(frame_dir)
        if mix_id is None:
            continue
        seq_name = f"{args.seq_prefix}{mix_id}"

        dst_frame = frame_out / f"{seq_name}.mp4"
        if frame_dir.resolve() != dst_frame.resolve():
            safe_link(frame_dir, dst_frame, overwrite=args.overwrite)
            made_frames += 1

        src_audio = track2_root / "foa_eval" / f"{mix_id}.wav"
        if src_audio.exists():
            safe_link(src_audio, audio_out / f"{seq_name}.wav", overwrite=args.overwrite)
            made_audio += 1
        else:
            missing_audio += 1

        meta_path = meta_out / f"{seq_name}.csv"
        if args.overwrite or not meta_path.exists():
            meta_path.write_text("", encoding="utf-8")
            made_meta += 1

    print(
        "[track2-shadow] "
        f"frames={len(frame_dirs)} linked_frames={made_frames} "
        f"audio_links={made_audio} metadata={made_meta} missing_audio={missing_audio} "
        f"shadow={shadow}"
    )


if __name__ == "__main__":
    main()
