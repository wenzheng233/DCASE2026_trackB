#!/usr/bin/env python3
"""Build DCASE2026 Task3B submission packages from selected blind outputs.

The official example uses this layout:

  task3/<Author>_<Affiliation>_task3.technical_report.pdf
  task3/<Label>/<Label>.meta.yaml
  task3/<Label>/<Label>/mix001.json

This script creates both one four-system package and one package per system.
It also normalises blind filenames from fold9_room0_mix001_inference.json to
mix001.json and strips local-only fields such as instance_id.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import textwrap
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path("/home/yishuoyang/AV-SELD/v16_23prior")
OUT_ROOT = ROOT / "official_submissions" / "20260616_task3b_ver1_ver2_ver3_ver4"
ZIP_ROOT = ROOT / "official_submissions" / "zips"

AUTHOR_LASTNAME = "Yang"
AUTHOR_FIRSTNAME = "Yishuo"
AUTHOR_EMAIL = "2021302121303@whu.edu.cn"
AFFILIATION_ABBR = "AVSELD"
AFFILIATION_INSTITUTE = "AV-SELD local submission team"
AFFILIATION_DEPARTMENT = "!!null"
AFFILIATION_LOCATION = "!!null"

TECH_REPORT_NAME = f"{AUTHOR_LASTNAME}_{AFFILIATION_ABBR}_task3.technical_report.pdf"


@dataclass(frozen=True)
class Candidate:
    label: str
    name: str
    abbreviation: str
    source_dir: Path
    dev_macro: float
    dev_micro: float
    dev_macro_ap50: float
    dev_micro_ap50: float
    method_summary: str
    post_processing: str


CANDIDATES = [
    Candidate(
        label="ver1",
        name="ver1",
        abbreviation="ver1",
        source_dir=ROOT
        / "blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_tiebreak_walkfirst_energyasc_track2_blind/inference_outputs",
        dev_macro=0.08308650,
        dev_micro=0.07924123,
        dev_macro_ap50=0.086135,
        dev_micro_ap50=0.086465,
        method_summary=(
            "v16 audiovisual proposal masks with same-input 2023 AV-SELD teacher "
            "category prior, per-class count prior, class-6 score gate, selected "
            "class pruning, and deterministic score tie-breaks with lower-energy "
            "masks ranked earlier inside tied priority groups."
        ),
        post_processing=(
            "2023 teacher prior threshold 14.5 deg; matched detections score 1.0; "
            "per-class count cap; class 6 uncapped with score >= 0.22; dropped "
            "classes 3,5,7,8,11; class priority 6,1,8,0,9,5,4,10,3,2,11,7,12; "
            "energy-ascending inner tie-break for score ties."
        ),
    ),
    Candidate(
        label="ver2",
        name="ver2",
        abbreviation="ver2",
        source_dir=ROOT
        / "blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_tiebreak_walkfirst_track2_blind/inference_outputs",
        dev_macro=0.08321187,
        dev_micro=0.07625635,
        dev_macro_ap50=0.086482,
        dev_micro_ap50=0.085867,
        method_summary=(
            "v16 audiovisual proposal masks with same-input 2023 AV-SELD teacher "
            "category prior, per-class count prior, class-6 score gate, selected "
            "class pruning, and deterministic walk-first score tie-breaks."
        ),
        post_processing=(
            "2023 teacher prior threshold 14.5 deg; matched detections score 1.0; "
            "per-class count cap; class 6 uncapped with score >= 0.22; dropped "
            "classes 3,5,7,8,11; class priority 6,1,8,0,9,5,4,10,3,2,11,7,12."
        ),
    ),
    Candidate(
        label="ver3",
        name="ver3",
        abbreviation="ver3",
        source_dir=ROOT
        / "blind/v16_23same_thr14p75_score1_count_per_class_keep_1_track2_blind/inference_outputs",
        dev_macro=0.15075013,
        dev_micro=0.04172099,
        dev_macro_ap50=0.187957,
        dev_micro_ap50=0.050796,
        method_summary=(
            "Macro-oriented v16 plus same-input 2023 teacher prior variant "
            "that keeps only Male speech predictions after per-class count prior."
        ),
        post_processing=(
            "2023 teacher prior threshold 14.75 deg; matched detections score 1.0; "
            "per-class count cap; keep only category 1."
        ),
    ),
    Candidate(
        label="ver4",
        name="ver4",
        abbreviation="ver4",
        source_dir=ROOT / "blind/nonclean_v14_raw_track2_blind/merged_preds",
        dev_macro=0.17580016,
        dev_micro=0.16672519,
        dev_macro_ap50=0.0,
        dev_micro_ap50=0.0,
        method_summary=(
            "Historical v14 raw blind inference variant."
        ),
        post_processing="Raw v14 blind prediction JSONs.",
    ),
]


def escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_minimal_pdf(path: Path, title: str, lines: Iterable[str]) -> None:
    wrapped: list[str] = [title, ""]
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=92) or [""])

    content_lines = ["BT", "/F1 10 Tf", "72 780 Td", "14 TL"]
    for line in wrapped[:48]:
        content_lines.append(f"({escape_pdf_text(line)}) Tj")
        content_lines.append("T*")
    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("ascii")

    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream"
    )

    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{idx} 0 obj\n".encode("ascii"))
        data.extend(obj)
        data.extend(b"\nendobj\n")
    xref_pos = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    data.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        data.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    data.extend(
        b"trailer\n"
        + f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode("ascii")
        + b"startxref\n"
        + str(xref_pos).encode("ascii")
        + b"\n%%EOF\n"
    )
    path.write_bytes(data)


def normalised_name(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_inference"):
        stem = stem[: -len("_inference")]
    match = re.search(r"(mix\d+)$", stem)
    if not match:
        raise ValueError(f"Cannot infer mix name from {path.name}")
    return f"{match.group(1)}.json"


def validate_and_normalise_payload(path: Path) -> tuple[dict, dict]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    annotations = data.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError(f"{path} does not contain an annotations list")

    class_counts: dict[int, int] = {}
    max_score = 0.0
    for ann in annotations:
        for key in ("metadata_frame_index", "category_id", "score", "segmentation"):
            if key not in ann:
                raise ValueError(f"{path} annotation missing {key}")
        ann.pop("instance_id", None)
        category = int(ann["category_id"])
        if not 0 <= category <= 12:
            raise ValueError(f"{path} invalid category_id {category}")
        ann["category_id"] = category
        ann["metadata_frame_index"] = int(ann["metadata_frame_index"])
        ann["score"] = float(ann["score"])
        if not 0.0 <= ann["score"] <= 1.0:
            raise ValueError(f"{path} invalid score {ann['score']}")
        max_score = max(max_score, ann["score"])
        class_counts[category] = class_counts.get(category, 0) + 1
        if "distance" in ann:
            ann["distance"] = float(ann["distance"])
        segmentation = ann["segmentation"]
        if not isinstance(segmentation, list):
            raise ValueError(f"{path} invalid segmentation container")
        for poly in segmentation:
            if not isinstance(poly, list):
                raise ValueError(f"{path} invalid polygon container")
            for triplet in poly:
                if not isinstance(triplet, list) or len(triplet) != 3:
                    raise ValueError(f"{path} invalid segmentation triplet")
                x, y, intensity = map(float, triplet)
                if not 0.0 <= x <= 360.0:
                    raise ValueError(f"{path} x out of range: {x}")
                if not 0.0 <= y <= 180.0:
                    raise ValueError(f"{path} y out of range: {y}")
                triplet[0], triplet[1], triplet[2] = x, y, intensity
    stats = {
        "annotations": len(annotations),
        "class_counts": class_counts,
        "max_score": max_score,
    }
    return {"annotations": annotations}, stats


def write_meta_yaml(path: Path, candidate: Candidate) -> None:
    path.write_text(
        f"""# Submission information
submission:
  label: {candidate.label}
  name: {candidate.name}
  abbreviation: {candidate.abbreviation}
  authors:
    - lastname: {AUTHOR_LASTNAME}
      firstname: {AUTHOR_FIRSTNAME}
      email: {AUTHOR_EMAIL}
      corresponding: true
      affiliation:
        abbreviation: {AFFILIATION_ABBR}
        institute: {AFFILIATION_INSTITUTE}
        department: {AFFILIATION_DEPARTMENT}
        location: {AFFILIATION_LOCATION}

system:
  description:
    model_type: Audiovisual
    input_format: MIC
    input_sampling_rate: 24kHz
    acoustic_features: log mel spectra, learned spatial acoustic image masks
    visual_features: RGB video frames, ResNet/Conformer audiovisual features
    data_augmentation: weak-class sampling, frame/audio context sampling
    machine_learning_method: ResNet-Conformer mask proposal model, 2023 AV-SELD teacher prior, deterministic post-processing
    external_datasets: DCASE2023 Task3 development data for the teacher prior
    pre_trained_models: 2023 full ResNet-Conformer AV-SELD checkpoint model_007
    post_processing: {candidate.post_processing}
  complexity:
    total_parameters: 100000000
  source_code: https://github.com/iranroman/DCASE2026_Task3_SAISELD_baseline

results:
  development_dataset:
    overall:
      mAP: {candidate.dev_macro:.8f}
      EFRQ: !!null
""",
        encoding="utf-8",
    )


def build_system_dir(base: Path, candidate: Candidate) -> dict:
    system_root = base / "task3" / candidate.label
    pred_out = system_root / candidate.label
    pred_out.mkdir(parents=True, exist_ok=True)
    write_meta_yaml(system_root / f"{candidate.label}.meta.yaml", candidate)

    files = sorted(candidate.source_dir.glob("*.json"))
    if len(files) != 79:
        raise ValueError(f"{candidate.label}: expected 79 JSONs, got {len(files)} in {candidate.source_dir}")

    seen = set()
    total_annotations = 0
    max_json_bytes = 0
    max_score = 0.0
    class_counts: dict[int, int] = {}
    for src in files:
        name = normalised_name(src)
        if name in seen:
            raise ValueError(f"{candidate.label}: duplicate output name {name}")
        seen.add(name)
        payload, stats = validate_and_normalise_payload(src)
        out_path = pred_out / name
        out_path.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=True), encoding="utf-8")
        size = out_path.stat().st_size
        if size >= 20 * 1024 * 1024:
            raise ValueError(f"{candidate.label}: {name} is {size} bytes, above the 20 MB limit")
        max_json_bytes = max(max_json_bytes, size)
        total_annotations += stats["annotations"]
        max_score = max(max_score, stats["max_score"])
        for cls, count in stats["class_counts"].items():
            class_counts[cls] = class_counts.get(cls, 0) + count

    expected = {f"mix{i:03d}.json" for i in range(1, 80)}
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(f"{candidate.label}: bad mix set, missing={missing}, extra={extra}")

    return {
        "label": candidate.label,
        "name": candidate.name,
        "json_files": len(files),
        "total_annotations": total_annotations,
        "max_json_bytes": max_json_bytes,
        "max_score": max_score,
        "class_counts": {str(k): v for k, v in sorted(class_counts.items())},
        "dev_macro_mAP": candidate.dev_macro,
        "dev_micro_mAP": candidate.dev_micro,
    }


def zip_dir(src_dir: Path, zip_path: Path, include_root_files: bool = False) -> str:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                rel = path.relative_to(src_dir).as_posix()
                if not include_root_files and "/" not in rel:
                    continue
                info = zipfile.ZipInfo(rel)
                info.date_time = (2026, 6, 16, 16, 0, 0)
                info.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(info, path.read_bytes(), compresslevel=9)
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    return digest


def build_readme(manifest: dict) -> str:
    lines = [
        "# DCASE2026 Task3B Official Submission Package",
        "",
        "Generated from `/home/yishuoyang/AV-SELD/v16_23prior`.",
        "",
        "## Systems",
        "",
        "| label | dev macro mAP | dev micro mAP | JSONs | max JSON bytes | annotations |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in manifest["systems"]:
        lines.append(
            f"| {row['label']} | {row['dev_macro_mAP']:.8f} | "
            f"{row['dev_micro_mAP']:.8f} | {row['json_files']} | {row['max_json_bytes']} | "
            f"{row['total_annotations']} |"
        )
    lines.extend(
        [
            "",
            "## Package Layout",
            "",
            "Each system follows the official example layout:",
            "",
            "```text",
            "task3/Yang_AVSELD_task3.technical_report.pdf",
            "task3/verN/verN.meta.yaml",
            "task3/verN/verN/mix001.json ... mix079.json",
            "```",
            "",
            "The local source JSON names were converted from `fold9_room0_mixNNN_inference.json` to `mixNNN.json`.",
            "All JSON files are below the 20 MB per-file limit and contain a top-level `annotations` list.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True)
    ZIP_ROOT.mkdir(parents=True, exist_ok=True)
    for old_zip in ZIP_ROOT.glob("*.zip"):
        old_zip.unlink()
    for old_sha in ZIP_ROOT.glob("SHA256SUMS*.txt"):
        old_sha.unlink()

    task3_dir = OUT_ROOT / "task3"
    task3_dir.mkdir(parents=True)
    write_minimal_pdf(
        task3_dir / TECH_REPORT_NAME,
        "DCASE2026 Task3B AV-SELD submission",
        [
            "This package contains four Task3B audiovisual systems named ver1, ver2, ver3, and ver4.",
            "The variants use audiovisual proposal masks, 2023 AV-SELD teacher priors, count priors, deterministic post-processing, or the historical v14 prediction branch.",
            "Prediction JSON files follow the official Task3 example with a top-level annotations list and mix001.json to mix079.json naming.",
        ],
    )

    manifest = {
        "created": "2026-06-16",
        "package_root": str(OUT_ROOT),
        "systems": [],
        "zips": {},
    }

    for candidate in CANDIDATES:
        manifest["systems"].append(build_system_dir(OUT_ROOT, candidate))

    (OUT_ROOT / "README_20260616.md").write_text(build_readme(manifest), encoding="utf-8")
    (OUT_ROOT / "manifest.pre_zip.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    all_zip = ZIP_ROOT / "ver1_ver2_ver3_ver4.zip"
    manifest["zips"][all_zip.name] = {
        "path": str(all_zip),
        "sha256": zip_dir(OUT_ROOT, all_zip),
        "systems": [c.label for c in CANDIDATES],
        "strict_official_layout": True,
    }

    for candidate in CANDIDATES:
        single_root = OUT_ROOT.parent / f"20260616_{candidate.label}_single"
        if single_root.exists():
            shutil.rmtree(single_root)
        single_task3 = single_root / "task3"
        single_task3.mkdir(parents=True)
        shutil.copy2(task3_dir / TECH_REPORT_NAME, single_task3 / TECH_REPORT_NAME)
        shutil.copytree(OUT_ROOT / "task3" / candidate.label, single_task3 / candidate.label)
        zip_path = ZIP_ROOT / f"{candidate.label}.zip"
        manifest["zips"][zip_path.name] = {
            "path": str(zip_path),
            "sha256": zip_dir(single_root, zip_path),
            "systems": [candidate.label],
            "strict_official_layout": True,
        }

    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    sha_lines = []
    for name, info in sorted(manifest["zips"].items()):
        sha_lines.append(f"{info['sha256']}  {name}")
    (ZIP_ROOT / "SHA256SUMS_official_task3b_20260616.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
