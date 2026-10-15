# DCASE 2026 Task 3B: Audio-Visual SELD

This repository contains the reproducibility snapshot and official submission files for our DCASE 2026 Task 3B audio-visual sound event localization and detection (AV-SELD) work. The main clean submission pipeline combines v16 event and localization predictions with same-input predictions from a 2023 teacher, followed by score and per-class count post-processing.

## Repository contents

```text
source_code/          Model, inference, evaluation, and prior/post-processing code
scripts/              Reproduction, blind inference, and submission packaging scripts
official_submission/  Four Task 3B submission systems and their manifest
metrics/              Development-set metric summaries and experiment results
checkpoints/          Checkpoint directory (model weights are not tracked)
logs/                 Selected run logs
model_config.json     v16 model configuration snapshot
REPRODUCE.md          Detailed reproduction notes and blind inference workflow
ARTIFACTS.md          Artifact inventory and provenance
NEXT_STEPS.md         Follow-up experiment notes
```

Large generated prediction folders and submission archives are intentionally excluded. Model weights are also excluded; see [`.gitignore`](.gitignore). The official submission package is available under [`official_submission/task3`](official_submission/task3), with system metadata and development metrics in [`official_submission/manifest.json`](official_submission/manifest.json).

## Submitted systems

The values below are development-set metrics recorded in the submission manifest. Each system contains 79 prediction JSON files.

| System | Status | Dev macro mAP | Dev micro mAP | Description |
| --- | --- | ---: | ---: | --- |
| `Yang_AVSELD_task3b_1` | clean | 0.08308650 | 0.07924123 | v16 + 2023 prior, micro-oriented score tie-break |
| `Yang_AVSELD_task3b_2` | clean | 0.08321187 | 0.07625635 | v16 + 2023 prior, micro-oriented candidate |
| `Yang_AVSELD_task3b_3` | clean | 0.15075013 | 0.04172099 | Macro-oriented candidate retaining Male speech predictions |
| `Yang_AVSELD_task3b_4` | non-clean | 0.17580016 | 0.16672519 | v14 raw candidate; marked non-clean in the manifest |

These are development-set results, not blind test scores. System 4 is explicitly marked non-clean and should not be presented as a clean submission.

## Reproduction

Start with [`REPRODUCE.md`](REPRODUCE.md) for the recorded v16 + same-input 2023 prior workflow, evaluation commands, and Track 2 blind inference steps. The scripts in this snapshot were prepared in a specific Linux environment; update their data, checkpoint, Python, and project paths for your environment before running them.

The repository does not include model checkpoints, the DCASE datasets, cached predictions, or the 2023 teacher outputs. Reproducing inference or evaluation therefore requires obtaining the corresponding model weights and data separately and placing/configuring them as expected by the scripts. The checkpoint filenames and artifact provenance are documented in [`ARTIFACTS.md`](ARTIFACTS.md).

For a quick inventory of the included source files, see [`files_snapshot.txt`](files_snapshot.txt). Submission package layout and per-system details are documented in [`official_submission/README_20260616.md`](official_submission/README_20260616.md).
