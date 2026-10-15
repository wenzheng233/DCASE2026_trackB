#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/yishuoyang/AV-SELD/DCASE2026_Task3_SAISELD_baseline}"
PY="${PYTHON_BIN:-/home/yishuoyang/anaconda3/envs/audio_visual_seld_dcase2023/bin/python}"

BASE_PRED="${BASE_PRED:-${ROOT}/experiments/official_eval_v16_clean_epoch1_full_20260611_g15/merged_preds}"
TEACHER="${TEACHER:-/home/yishuoyang/AV-SELD/output/full_resnet_conformer_e20_20260402_234300_best7_eval_20260415/0.000050}"
GT_DIR="${GT_DIR:-/home/yishuoyang/dataset/decase_challenge/STAIRS26_ACS_20260426/labels_dev}"
FRAMES_BASE="${FRAMES_BASE:-/home/yishuoyang/dataset/decase_challenge/STAIRS26_ACS_20260426/frames_dev}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/experiments/prior23_sameinput_fast_20260615}"
NUM_WORKERS="${NUM_WORKERS:-4}"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

PRIOR_OUT="${OUT_ROOT}/v16_23same_thr14p5_score1_stairs_compact"
MICRO_OUT="${OUT_ROOT}/v16_23same_thr14p5_score1_count_per_class_uncap_10_11_12_compact"
MACRO_OUT="${OUT_ROOT}/v16_23same_thr15_score1_count_per_class_keeporder_compact"

run_cmd() {
  echo "+ $*"
  if [[ "${DCASE2026_DRY_RUN:-0}" != "1" ]]; then
    "$@"
  fi
}

cd "${ROOT}"

run_cmd "${PY}" apply_2023_prior_to_26_predictions.py \
  --pred-dir "${BASE_PRED}" \
  --teacher-csv-dir "${TEACHER}" \
  --out-dir "${PRIOR_OUT}" \
  --match-thr-deg 14.5 \
  --score-mode matched_one \
  --coord-convention stairs26

run_cmd "${PY}" evaluate.py \
  --pred_dir "${PRIOR_OUT}/inference_outputs" \
  --gt_dir "${GT_DIR}" \
  --frames_base "${FRAMES_BASE}" \
  --output_dir "${PRIOR_OUT}/eval_full_w4_threads1" \
  --n_comp_seqs 0 \
  --num_workers "${NUM_WORKERS}"

run_cmd "${PY}" apply_2023_count_prior_to_26_predictions.py \
  --pred-dir "${PRIOR_OUT}/inference_outputs" \
  --teacher-csv-dir "${TEACHER}" \
  --out-dir "${MICRO_OUT}" \
  --count-mode per_class \
  --uncapped-classes "10,11,12"

run_cmd "${PY}" evaluate.py \
  --pred_dir "${MICRO_OUT}/inference_outputs" \
  --gt_dir "${GT_DIR}" \
  --frames_base "${FRAMES_BASE}" \
  --output_dir "${MICRO_OUT}/eval_full_w4_threads1" \
  --n_comp_seqs 0 \
  --num_workers "${NUM_WORKERS}"

run_cmd "${PY}" apply_2023_count_prior_to_26_predictions.py \
  --pred-dir "${PRIOR_OUT}/inference_outputs" \
  --teacher-csv-dir "${TEACHER}" \
  --out-dir "${MACRO_OUT}" \
  --count-mode per_class \
  --keep-order

run_cmd "${PY}" evaluate.py \
  --pred_dir "${MACRO_OUT}/inference_outputs" \
  --gt_dir "${GT_DIR}" \
  --frames_base "${FRAMES_BASE}" \
  --output_dir "${MACRO_OUT}/eval_full_w4_threads1" \
  --n_comp_seqs 0 \
  --num_workers "${NUM_WORKERS}"

echo "[done] prior metrics: ${PRIOR_OUT}/eval_full_w4_threads1/metrics.json"
echo "[done] micro metrics: ${MICRO_OUT}/eval_full_w4_threads1/metrics.json"
echo "[done] macro metrics: ${MACRO_OUT}/eval_full_w4_threads1/metrics.json"
