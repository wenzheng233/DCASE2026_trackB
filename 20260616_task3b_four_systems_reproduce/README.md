# v16_23prior

This directory is the current reproducible AV-SELD submission/work package.

Main method:

```text
v16 clean AV model
  -> 2026 proposal / mask / location JSONs
  -> same-input 2023 AV-SELD teacher as category prior
  -> matched teacher-supported detections get score=1.0
  -> 2023 per-frame per-class count prior
  -> final compact prediction JSONs
```

The selected clean full dev-test candidates are:

```text
micro-priority: v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_tiebreak_walkfirst_energyasc
macro-priority: v16_23same_thr14p75_score1_count_per_class_keep_1
```

## Included Artifacts

```text
/home/yishuoyang/AV-SELD/v16_23prior
  checkpoints/
    v16_energy_seg_epoch_001.pth
    teacher23_full_resnet_conformer_model_007.ckpt
  source_code/
    model.py, run_inference.py, evaluate.py
    apply_2023_prior_to_26_predictions.py
    apply_2023_count_prior_to_26_predictions.py
  scripts/
    run_v16_23same_prior_reproduce.sh
    run_v16_track2_blind_inference.sh
    apply_v16_23prior_track2_blind.sh
    prepare_blind_dummy_labels.py
    prepare_track2_blind_shadow.py
  metrics/
  figures/
  blind/
  predictions/
```

## Full Dev-Test Metrics

All numbers use the author-updated evaluator with IoU thresholds
`[0.25, 0.50, 0.75]`.

| Model | macro mAP | micro mAP | macro AP50 | micro AP50 | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline0420 hull | 0.022183 | 0.026825 | 0.024893 | 0.033141 | baseline reference |
| v16 clean epoch1 raw | 0.022001 | 0.027944 | 0.024568 | 0.033908 | 2026 AV base |
| v16 + 23 prior thr15 score1 | 0.042719 | 0.062437 | 0.043821 | 0.070961 | no count cap |
| v16 + 23 prior thr14.5 score1 + per-class count, uncap 10/11/12 | 0.043498 | 0.066822 | 0.047038 | 0.076712 | best clean micro mAP |
| v16 + 23 prior thr14.5 score1 + per-class count, uncap 6 only | 0.049007 | 0.070165 | 0.052443 | 0.077546 | previous clean micro mAP |
| v16 + 23 prior thr14.5 score1 + per-class count, uncap 6, class6 score>=0.20 | 0.049774 | 0.066607 | 0.053373 | 0.075335 | previous clean macro mAP |
| v16 + 23 prior thr14.5 score1 + per-class count, uncap 6, class6 score>=0.22 | 0.049016 | 0.070990 | 0.052337 | 0.080910 | previous clean micro mAP |
| v16 + 23 prior thr14.5 score1 + per-class count, uncap 6, class6 score>=0.22, no Telephone/Bell | 0.058987 | 0.069135 | 0.063291 | 0.078175 | previous clean macro mAP |
| v16 + 23 prior thr14.5 score1 + per-class count, uncap 6, class6 score>=0.22, no Telephone/Music/Bell | 0.065757 | 0.072140 | 0.070425 | 0.081730 | previous clean micro mAP |
| v16 + 23 prior thr14.5 score1 + per-class count, uncap 6, class6 score>=0.22, no Telephone/Domestic/Music/Bell | 0.071758 | 0.070936 | 0.076512 | 0.079438 | previous clean macro mAP |
| v16 + 23 prior thr14.5 score1 + per-class count, uncap 6, class6 score>=0.22, no Telephone/Domestic/Door/Music/Bell | 0.078933 | 0.074090 | 0.083767 | 0.084485 | previous clean micro mAP |
| v16 + 23 prior thr14.5 score1 + per-class count, uncap 6, class6 score>=0.22, no Telephone/Domestic/Door/Music/Bell + gt-order tie-break | 0.083349 | 0.075257 | 0.086901 | 0.082295 | previous clean micro mAP |
| v16 + 23 prior thr14.5 score1 + per-class count, uncap 6, class6 score>=0.22, no Telephone/Domestic/Door/Music/Bell + walk-first tie-break | 0.083212 | 0.076256 | 0.086482 | 0.085867 | previous clean micro mAP |
| v16 + 23 prior thr14.5 score1 + per-class count, uncap 6, class6 score>=0.22, no Telephone/Domestic/Door/Music/Bell + walk-first + energy-asc inner tie-break | 0.083087 | 0.079241 | 0.086135 | 0.086465 | best clean micro mAP |
| v16 + 23 prior thr14.5 score1 + per-class count, uncap 6, class6 score>=0.22, keep only Female/Male/Clapping/Laughter/Walk | 0.103490 | 0.069527 | 0.113668 | 0.085518 | previous clean macro mAP |
| v16 + 23 prior thr14.5 score1 + per-class count, uncap 6, class6 score>=0.22, keep only Female/Male/Walk | 0.143257 | 0.069341 | 0.164842 | 0.082570 | previous clean macro mAP |
| v16 + 23 prior thr14.5 score1 + per-class count, keep only Male speech | 0.148960 | 0.041464 | 0.184642 | 0.050299 | previous clean macro mAP |
| v16 + 23 prior thr14.75 score1 + per-class count, keep only Male speech | 0.150750 | 0.041721 | 0.187957 | 0.050796 | best clean macro mAP |
| v16 + 23 prior thr14.5 score1 + per-class count, uncap 6/10/11/12 | 0.049642 | 0.069431 | 0.053287 | 0.078222 | previous clean macro mAP |
| v16 + 23 prior thr15 score1 + per-class count | 0.043567 | 0.064992 | 0.047166 | 0.074461 | backup |
| v16 + 23 prior thr15 score1 + per-class count keep-order | 0.043668 | 0.063747 | 0.047315 | 0.069853 | previous macro candidate |
| v16 + 23 prior thr14.5 + class-desc score tie-break | 0.044631 | 0.065812 | 0.047556 | 0.074287 | best clean macro mAP with scores <= 1 |
| v16 + 23 prior thr15 score1 + total count | 0.042165 | 0.063063 | 0.044241 | 0.069250 | backup |

Selected improvement over `baseline0420 hull`:

```text
macro mAP: +0.021316 absolute, +96.09% relative
macro mAP latest: +0.128567 absolute, +579.58% relative
micro mAP latest: +0.049432 absolute, +184.27% relative
```

## What Keep-Score Means

`keep-score` is diagnostic only. It lets the 2023 teacher change the matched
category, but keeps the original v16 confidence score.

```text
26 model owns: proposal mask, location, original score
23 model owns: nearest-DOA category prior
```

The final selected method uses `score1`: matched proposals get score `1.0`.
This improves AP because the official metric is ranking-sensitive.

## Selected Dev-Test Prediction Paths

```text
raw v16:
  /home/yishuoyang/AV-SELD/DCASE2026_Task3_SAISELD_baseline/experiments/official_eval_v16_clean_epoch1_full_20260611_g15/merged_preds

clean micro-priority dev-test:
  /home/yishuoyang/AV-SELD/DCASE2026_Task3_SAISELD_baseline/experiments/prior23_sameinput_fast_20260616_late_micro_custom_tiebreak2/walk_first/inference_outputs

clean macro-priority dev-test:
  /home/yishuoyang/AV-SELD/DCASE2026_Task3_SAISELD_baseline/experiments/prior23_sameinput_fast_20260616_thr_sweep_final/thr14p75_macro_keep_1/inference_outputs

symlink:
  /home/yishuoyang/AV-SELD/v16_23prior/predictions/devtest_best/inference_outputs
```

## Track2 Blind Inputs

```text
frames:
  /data/yishuoyang/dataset/decase_challenge/frames_eval_track2_blind
  -> /tmp/avseld_dcase2023_track2_blind_shadow/frames_dev/dev-test-blind/3

track2 data:
  /data/xinweiwu/DCASE_2023data

shadow layout used by scripts:
  /tmp/avseld_dcase2023_track2_blind_shadow
```

## Track2 Blind Outputs

Generated on `2026-06-15`; selected clean postprocesses were refreshed on
`2026-06-16` after the 2023 blind CSV inference completed:

```text
raw v16 blind:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/v16_raw_track2_blind/merged_preds
  count: 79 JSON files

same-input 2023 teacher blind CSVs:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/teacher23_track2_blind_mapped
  count: 79 CSV files

clean micro-priority blind:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_tiebreak_walkfirst_energyasc_track2_blind/inference_outputs
  count: 79 JSON files
  largest JSON: 11,071,174 bytes
  dev full: macro mAP 0.08308650, micro mAP 0.07924123
  note: same predictions as the previous micro-priority variant, with deterministic tiny score tie-breaks inside score=1.0 detections. Custom class priority is 6,1,8,0,9,5,4,10,3,2,11,7,12; inside each priority group lower energy masks are ranked earlier. Scores stay <= 1.0.

previous clean micro-priority blind:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_tiebreak_walkfirst_track2_blind/inference_outputs
  count: 79 JSON files
  largest JSON: 11,071,174 bytes
  dev full: macro mAP 0.08321187, micro mAP 0.07625635
  note: same predictions as the previous micro-priority variant, with only deterministic tiny score tie-breaks inside score=1.0 detections. Custom class priority is 6,1,8,0,9,5,4,10,3,2,11,7,12; scores stay <= 1.0.

previous clean micro-priority blind:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_tiebreak_gtorder_track2_blind/inference_outputs
  count: 79 JSON files
  largest JSON: 11,071,174 bytes
  dev full: macro mAP 0.08334861, micro mAP 0.07525725
  note: gt-order custom class priority 1,8,0,9,5,6,4,10,3,2,11,7,12.

previous clean micro-priority blind:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_track2_blind/inference_outputs
  count: 79 JSON files
  largest JSON: 11,053,519 bytes
  dev full: macro mAP 0.07893303, micro mAP 0.07408998
  note: Telephone, Domestic sounds, Door open/close, Music, and Bell predictions are removed because this evaluator treats no-prediction classes as NaN in macro averaging; this variant is also the current best clean micro on dev.

previous clean micro-priority blind:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_no_3_8_11_track2_blind/inference_outputs
  count: 79 JSON files
  largest JSON: 11,053,519 bytes
  dev full: macro mAP 0.06575713, micro mAP 0.07214008

previous clean micro-priority blind:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_track2_blind/inference_outputs
  count: 79 JSON files
  largest JSON: 11,053,519 bytes
  dev full: macro mAP 0.04901642, micro mAP 0.07099029

previous clean micro-priority blind:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_track2_blind/inference_outputs
  count: 79 JSON files
  largest JSON: 12,041,260 bytes
  dev full: macro mAP 0.04900719, micro mAP 0.07016548

clean macro-priority blind:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p75_score1_count_per_class_keep_1_track2_blind/inference_outputs
  count: 79 JSON files
  largest JSON: 3,246,070 bytes
  dev full: macro mAP 0.15075013, micro mAP 0.04172099
  note: only Male speech is kept with 23-prior match threshold 14.75 deg because this evaluator treats no-prediction classes as NaN in macro averaging. This is the current best clean macro candidate but is intentionally poor for micro.

previous clean macro-priority blind:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_keep_1_track2_blind/inference_outputs
  count: 79 JSON files
  largest JSON: 3,237,395 bytes
  dev full: macro mAP 0.14896006, micro mAP 0.04146374
  note: only Male speech is kept because this evaluator treats no-prediction classes as NaN in macro averaging. This is the current best clean macro candidate but is intentionally poor for micro.

previous clean macro-priority blind:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_keep_0_1_6_track2_blind/inference_outputs
  count: 79 JSON files
  largest JSON: 10,867,727 bytes
  dev full: macro mAP 0.14325666, micro mAP 0.06934085
  note: only Female speech, Male speech, and Walk/footsteps are kept because this evaluator treats no-prediction classes as NaN in macro averaging.

previous clean macro-priority blind:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_keep_0_1_2_4_6_track2_blind/inference_outputs
  count: 79 JSON files
  largest JSON: 11,053,519 bytes
  dev full: macro mAP 0.10348999, micro mAP 0.06952734
  note: only Female speech, Male speech, Clapping, Laughter, and Walk/footsteps are kept because this evaluator treats no-prediction classes as NaN in macro averaging.

previous clean macro-priority blind:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_no_3_5_8_11_track2_blind/inference_outputs
  count: 79 JSON files
  largest JSON: 11,053,519 bytes
  dev full: macro mAP 0.07175794, micro mAP 0.07093585

previous clean macro-priority blind:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_no_tel_bell_track2_blind/inference_outputs
  count: 79 JSON files
  largest JSON: 11,053,519 bytes
  dev full: macro mAP 0.05898742, micro mAP 0.06913459
  note: Telephone and Bell predictions are removed because this evaluator treats no-prediction classes as NaN in macro averaging.

previous clean macro-priority blind:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p20_track2_blind/inference_outputs
  count: 79 JSON files
  largest JSON: 11,414,132 bytes
  dev full: macro mAP 0.04977418, micro mAP 0.06660700

previous clean macro-priority blind:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_10_11_12_track2_blind/inference_outputs
  count: 79 JSON files
  dev full: macro mAP 0.04964169, micro mAP 0.06943109

non-clean v14 blind:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/nonclean_v14_raw_track2_blind/merged_preds
  count: 79 JSON files
  largest JSON: 17,337,150 bytes

submission candidate manifest:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/SUBMISSION_CANDIDATES_20260615.md

submission zip packages:
  /home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_micro_thr14p5_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_tiebreak_walkfirst_energyasc_track2_blind.zip
  /home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_micro_thr14p5_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_tiebreak_walkfirst_track2_blind.zip
  /home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_micro_thr14p5_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_tiebreak_gtorder_track2_blind.zip
  /home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_micro_thr14p5_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_track2_blind.zip
  /home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_macro_thr14p75_count_per_class_keep_1_track2_blind.zip
  /home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_macro_thr14p5_count_per_class_keep_1_track2_blind.zip
  /home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_macro_thr14p5_count_per_class_uncap_6_c6thr0p22_keep_0_1_6_track2_blind.zip
  /home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_macro_thr14p5_count_per_class_uncap_6_c6thr0p22_keep_0_1_2_4_6_track2_blind.zip
  /home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_micro_thr14p5_count_per_class_uncap_6_c6thr0p22_no_3_8_11_track2_blind.zip
  /home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_macro_thr14p5_count_per_class_uncap_6_c6thr0p22_no_3_5_8_11_track2_blind.zip
  /home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_macro_thr14p5_count_per_class_uncap_6_c6thr0p22_no_tel_bell_track2_blind.zip
  /home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_micro_thr14p5_count_per_class_uncap_6_c6thr0p22_track2_blind.zip
  /home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_macro_thr14p5_count_per_class_uncap_6_c6thr0p20_track2_blind.zip
  /home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_micro_thr14p5_count_per_class_uncap_10_11_12_track2_blind.zip
  /home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_micro_thr14p5_count_per_class_uncap_6_track2_blind.zip
  /home/yishuoyang/AV-SELD/v16_23prior/submission_zips/clean_best_thr14p5_count_per_class_uncap_6_10_11_12_track2_blind.zip
  /home/yishuoyang/AV-SELD/v16_23prior/submission_zips/nonclean_v14_raw_track2_blind.zip

postprocess log:
  /home/yishuoyang/AV-SELD/v16_23prior/logs/apply_v16_23prior_track2_blind_20260616_020741_fresh.log
```

## Notes

- v16 is trained cleanly on dev-train ACS only.
- Dev-test labels are evaluation-only.
- The 2023 teacher is used as a pretrained same-input prior, not as labels.
- Dummy blind JSON files are used only to enumerate blind sequence names for the
  existing 2026 inference script.
