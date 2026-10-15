# Reproduce v16_23prior

Use:

```bash
cd /home/yishuoyang/AV-SELD/DCASE2026_Task3_SAISELD_baseline
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

## Full Dev-Test Selected Candidate

Generate `thr15 score1`:

```bash
PY=/home/yishuoyang/anaconda3/envs/audio_visual_seld_dcase2023/bin/python
BASE=experiments/official_eval_v16_clean_epoch1_full_20260611_g15/merged_preds
TEACHER=/home/yishuoyang/AV-SELD/output/full_resnet_conformer_e20_20260402_234300_best7_eval_20260415/0.000050
OUT=experiments/prior23_sameinput_fast_20260615/v16_23same_thr15_score1_stairs_compact

$PY apply_2023_prior_to_26_predictions.py \
  --pred-dir "$BASE" \
  --teacher-csv-dir "$TEACHER" \
  --out-dir "$OUT" \
  --match-thr-deg 15 \
  --score-mode matched_one \
  --coord-convention stairs26
```

Apply per-class count prior:

```bash
FINAL=experiments/prior23_sameinput_fast_20260615/v16_23same_thr15_score1_count_per_class_compact

$PY apply_2023_count_prior_to_26_predictions.py \
  --pred-dir "$OUT/inference_outputs" \
  --teacher-csv-dir "$TEACHER" \
  --out-dir "$FINAL" \
  --count-mode per_class
```

Evaluate:

```bash
$PY evaluate.py \
  --pred_dir "$FINAL/inference_outputs" \
  --gt_dir /home/yishuoyang/dataset/decase_challenge/STAIRS26_ACS_20260426/labels_dev \
  --frames_base /home/yishuoyang/dataset/decase_challenge/STAIRS26_ACS_20260426/frames_dev \
  --output_dir "$FINAL/eval_full_w4_threads1" \
  --n_comp_seqs 0 \
  --num_workers 4
```

Expected:

```text
macro mAP  0.04356655
micro mAP  0.06499237
macro AP50 0.04716608
micro AP50 0.07446150
```

## Track2 Blind Inference

First run raw v16 blind inference:

```bash
bash /home/yishuoyang/AV-SELD/v16_23prior/scripts/run_v16_track2_blind_inference.sh
```

Then wait until the same-input 2023 teacher blind CSVs are present:

```bash
find /home/yishuoyang/AV-SELD/output/dcase2023_track2_blind_model007_20260615/0.000050 \
  -maxdepth 2 -type f -path '*/mix_names/*.csv' | wc -l
```

Expected count is `79`.

Apply the selected prior and count post-processing:

```bash
bash /home/yishuoyang/AV-SELD/v16_23prior/scripts/apply_v16_23prior_track2_blind.sh
```

Final blind prediction directory:

```text
/home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr15_score1_count_per_class_track2_blind/inference_outputs
```

Generated `2026-06-15` verification:

```text
raw v16 merged count: 79
teacher blind CSV count: 79
final blind JSON count: 79
largest final JSON: 6,466,818 bytes
```

## Verify Package

```bash
sha256sum -c /home/yishuoyang/AV-SELD/v16_23prior/checksums.sha256
find /home/yishuoyang/AV-SELD/v16_23prior -maxdepth 4 -type f | sort
```
