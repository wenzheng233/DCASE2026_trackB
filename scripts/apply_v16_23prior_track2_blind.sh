#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/yishuoyang/AV-SELD/DCASE2026_Task3_SAISELD_baseline}"
PKG="${PKG:-/home/yishuoyang/AV-SELD/v16_23prior}"
PY="${PYTHON_BIN:-/home/yishuoyang/anaconda3/envs/audio_visual_seld_dcase2023/bin/python}"

RAW_PRED="${RAW_PRED:-${PKG}/blind/v16_raw_track2_blind/merged_preds}"
TEACHER_RAW="${TEACHER_RAW:-/home/yishuoyang/AV-SELD/output/dcase2023_track2_blind_model007_20260615/0.000050/mix_names}"
TEACHER_COPY="${TEACHER_COPY:-${PKG}/blind/teacher23_track2_blind_mix_names}"
TEACHER_MAPPED="${TEACHER_MAPPED:-${PKG}/blind/teacher23_track2_blind_mapped}"
PRIOR_OUT="${PRIOR_OUT:-${PKG}/blind/v16_23same_thr14p5_score1_track2_blind}"
FINAL_OUT="${FINAL_OUT:-${PKG}/blind/v16_23same_thr14p5_score1_count_per_class_uncap_10_11_12_track2_blind}"
if [[ -n "${COUNT_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  COUNT_EXTRA_ARGS_ARRAY=(${COUNT_EXTRA_ARGS})
else
  COUNT_EXTRA_ARGS_ARRAY=(--uncapped-classes "10,11,12")
fi

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "${ROOT}"

rm -rf "${TEACHER_COPY}" "${TEACHER_MAPPED}"
mkdir -p "${TEACHER_COPY}" "${TEACHER_MAPPED}"
find "${TEACHER_RAW}" -maxdepth 1 -type f -name '*.csv' -print0 | sort -z | while IFS= read -r -d '' src; do
  base="$(basename "${src}")"
  stem="${base%.csv}"
  cp -f "${src}" "${TEACHER_COPY}/${base}"
  if [[ "${stem}" =~ ^mix[0-9][0-9][0-9]$ ]]; then
    mapped="fold9_room0_${stem}.csv"
  else
    mapped="${base}"
  fi
  cp -f "${src}" "${TEACHER_MAPPED}/${mapped}"
done

TEACHER="${TEACHER_MAPPED}"
teacher_count="$(find "${TEACHER}" -maxdepth 1 -type f -name '*.csv' 2>/dev/null | wc -l)"
pred_count="$(find "${RAW_PRED}" -maxdepth 1 -type f -name '*_inference.json' 2>/dev/null | wc -l)"
echo "[blind-prior] teacher_raw=${TEACHER_RAW}"
echo "[blind-prior] teacher_csv=${teacher_count} raw_pred=${pred_count}"
if [[ "${teacher_count}" -lt 1 || "${pred_count}" -lt 1 ]]; then
  echo "[blind-prior] missing teacher CSVs or raw predictions" >&2
  exit 2
fi

"${PY}" apply_2023_prior_to_26_predictions.py \
  --pred-dir "${RAW_PRED}" \
  --teacher-csv-dir "${TEACHER}" \
  --out-dir "${PRIOR_OUT}" \
  --match-thr-deg 14.5 \
  --score-mode matched_one \
  --coord-convention stairs26

"${PY}" apply_2023_count_prior_to_26_predictions.py \
  --pred-dir "${PRIOR_OUT}/inference_outputs" \
  --teacher-csv-dir "${TEACHER}" \
  --out-dir "${FINAL_OUT}" \
  --count-mode per_class \
  "${COUNT_EXTRA_ARGS_ARRAY[@]}"

final_count="$(find "${FINAL_OUT}/inference_outputs" -maxdepth 1 -type f -name '*_inference.json' | wc -l)"
largest="$(find "${FINAL_OUT}/inference_outputs" -maxdepth 1 -type f -name '*_inference.json' -printf '%s %p\n' | sort -nr | head -1)"
echo "[blind-prior] final_predictions=${final_count}"
echo "[blind-prior] largest=${largest}"
echo "[blind-prior] DONE final=${FINAL_OUT}/inference_outputs"
