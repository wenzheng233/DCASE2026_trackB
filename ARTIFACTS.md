# Artifact map

## Core checkpoints

```text
v16 clean checkpoint:
  checkpoints/v16_energy_seg_epoch_001.pth
  original: /home/yishuoyang/AV-SELD/DCASE2026_Task3_SAISELD_baseline/experiments/formal_v16_clean_weakclass_from_fair_v11_20260610_1810_v16_clean_weak_small_g7_b1/epoch_checkpoints/energy_seg_epoch_001.pth

same-input 2023 teacher checkpoint:
  checkpoints/teacher23_full_resnet_conformer_model_007.ckpt
  original: /home/yishuoyang/AV-SELD/ckpt/full_resnet_conformer_e20_20260402_234300/0.000050/model_007.ckpt
```

## Prediction artifacts outside this release

The full prediction JSON directories are intentionally referenced, not copied.

```text
raw v16 clean predictions:
  /home/yishuoyang/AV-SELD/DCASE2026_Task3_SAISELD_baseline/experiments/official_eval_v16_clean_epoch1_full_20260611_g15/merged_preds

macro-best clean predictions:
  /home/yishuoyang/AV-SELD/DCASE2026_Task3_SAISELD_baseline/experiments/prior23_sameinput_fast_20260616_thr_sweep_final/thr14p75_macro_keep_1/inference_outputs

micro-best clean predictions:
  /home/yishuoyang/AV-SELD/DCASE2026_Task3_SAISELD_baseline/experiments/prior23_sameinput_fast_20260616_late_micro_custom_inner_tiebreak/custom_class_energy_asc/inference_outputs

micro-best clean compact predictions before score tie-break:
  /home/yishuoyang/AV-SELD/DCASE2026_Task3_SAISELD_baseline/experiments/prior23_sameinput_fast_20260616_class_prune_macro2/c6thr022_no_3_5_7_8_11/inference_outputs

same-input 2023 teacher CSV predictions:
  /home/yishuoyang/AV-SELD/output/full_resnet_conformer_e20_20260402_234300_best7_eval_20260415/0.000050
```

## Track2 blind artifacts

```text
raw v16 blind predictions:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/v16_raw_track2_blind/merged_preds
  count: 79 JSON files

same-input 2023 teacher blind CSV predictions:
  original: /home/yishuoyang/AV-SELD/output/dcase2023_track2_blind_model007_20260615/0.000050/mix_names
  mapped:   /home/yishuoyang/AV-SELD/v16_23prior/blind/teacher23_track2_blind_mapped
  count: 79 CSV files

clean micro-priority blind predictions:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_tiebreak_walkfirst_energyasc_track2_blind/inference_outputs
  count: 79 JSON files
  largest JSON: 11,071,174 bytes
  dev full: macro mAP 0.08308650, micro mAP 0.07924123
  note: current best clean micro candidate. It uses the compact variant below plus a deterministic tiny score tie-break with class priority 6,1,8,0,9,5,4,10,3,2,11,7,12 and lower-energy masks ranked earlier inside each priority group; scores stay <= 1.0.

previous clean micro-priority blind predictions:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_tiebreak_walkfirst_track2_blind/inference_outputs
  count: 79 JSON files
  largest JSON: 11,071,174 bytes
  dev full: macro mAP 0.08321187, micro mAP 0.07625635
  note: previous clean micro candidate. It uses the compact variant below plus a deterministic tiny score tie-break with class priority 6,1,8,0,9,5,4,10,3,2,11,7,12; scores stay <= 1.0.

previous clean micro-priority blind predictions:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_no_3_5_7_8_11_track2_blind/inference_outputs
  count: 79 JSON files
  largest JSON: 11,053,519 bytes
  dev full: macro mAP 0.07893303, micro mAP 0.07408998
  note: Telephone, Domestic sounds, Door open/close, Music, and Bell predictions are removed because this evaluator treats no-prediction classes as NaN in macro averaging.

previous clean micro-priority blind predictions:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_track2_blind/inference_outputs
  count: 79 JSON files
  largest JSON: 11,053,519 bytes
  dev full: macro mAP 0.04901642, micro mAP 0.07099029

previous clean micro-priority blind predictions:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_track2_blind/inference_outputs
  count: 79 JSON files
  largest JSON: 12,041,260 bytes
  dev full: macro mAP 0.04900719, micro mAP 0.07016548

clean macro-priority blind predictions:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p75_score1_count_per_class_keep_1_track2_blind/inference_outputs
  count: 79 JSON files
  largest JSON: 3,246,070 bytes
  dev full: macro mAP 0.15075013, micro mAP 0.04172099
  note: only Male speech is kept with 23-prior match threshold 14.75 deg because this evaluator treats no-prediction classes as NaN in macro averaging. This is the current best clean macro candidate but is intentionally poor for micro.

previous clean macro-priority blind predictions:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_keep_1_track2_blind/inference_outputs
  count: 79 JSON files
  largest JSON: 3,237,395 bytes
  dev full: macro mAP 0.14896006, micro mAP 0.04146374
  note: only Male speech is kept because this evaluator treats no-prediction classes as NaN in macro averaging. This is the current best clean macro candidate but is intentionally poor for micro.

previous clean macro-priority blind predictions:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_keep_0_1_6_track2_blind/inference_outputs
  count: 79 JSON files
  largest JSON: 10,867,727 bytes
  dev full: macro mAP 0.14325666, micro mAP 0.06934085
  note: only Female speech, Male speech, and Walk/footsteps are kept because this evaluator treats no-prediction classes as NaN in macro averaging.

previous clean macro-priority blind predictions:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_keep_0_1_2_4_6_track2_blind/inference_outputs
  count: 79 JSON files
  largest JSON: 11,053,519 bytes
  dev full: macro mAP 0.10348999, micro mAP 0.06952734
  note: Telephone, Domestic sounds, Music, and Bell predictions are removed because this evaluator treats no-prediction classes as NaN in macro averaging.

previous clean macro-priority blind predictions:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p22_no_tel_bell_track2_blind/inference_outputs
  count: 79 JSON files
  largest JSON: 11,053,519 bytes
  dev full: macro mAP 0.05898742, micro mAP 0.06913459
  note: Telephone and Bell predictions are removed because this evaluator treats no-prediction classes as NaN in macro averaging.

previous clean macro-priority blind predictions:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_c6thr0p20_track2_blind/inference_outputs
  count: 79 JSON files
  largest JSON: 11,414,132 bytes
  dev full: macro mAP 0.04977418, micro mAP 0.06660700

previous clean macro-priority blind predictions:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/v16_23same_thr14p5_score1_count_per_class_uncap_6_10_11_12_track2_blind/inference_outputs
  count: 79 JSON files
  dev full: macro mAP 0.04964169, micro mAP 0.06943109

non-clean v14 blind predictions:
  /home/yishuoyang/AV-SELD/v16_23prior/blind/nonclean_v14_raw_track2_blind/merged_preds
  count: 79 JSON files
  largest JSON: 17,337,150 bytes

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

logs:
  logs/run_v16_track2_blind_inference_tmux.log
  logs/apply_v16_23prior_track2_blind_20260616_020741_fresh.log
```

## Metric files

```text
metrics/full_metric_summary_20260615.tsv
metrics/baseline0420_hull_full_metrics.json
metrics/v16_clean_epoch1_full_metrics.json
metrics/v16_23same_thr20_keep_metrics.json
metrics/v16_23same_thr20_score1_metrics.json
metrics/v16_23same_thr20_score1_count_per_class_metrics.json
metrics/v16_23same_thr20_score1_count_total_metrics.json
metrics/v16_23same_thr30_score1_metrics.json
blind/SUBMISSION_CANDIDATES_20260615.md
```

## Source snapshots

```text
source_code/apply_2023_prior_to_26_predictions.py
source_code/apply_2023_count_prior_to_26_predictions.py
source_code/evaluate.py
source_code/run_inference.py
source_code/train.py
source_code/model.py
source_code/lam_model.py
source_code/fusion_blocks.py
scripts/run_v16_track2_blind_inference.sh
scripts/apply_v16_23prior_track2_blind.sh
scripts/prepare_track2_blind_shadow.py
scripts/prepare_blind_dummy_labels.py
```
