#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/yishuoyang/AV-SELD/DCASE2026_Task3_SAISELD_baseline}"
PKG="${PKG:-/home/yishuoyang/AV-SELD/v16_23prior}"
PY="${PYTHON_BIN:-/home/yishuoyang/anaconda3/envs/audio_visual_seld_dcase2023/bin/python}"

SHADOW="${SHADOW:-/tmp/avseld_dcase2023_track2_blind_shadow}"
TRACK2_ROOT="${TRACK2_ROOT:-/data/xinweiwu/DCASE_2023data}"
BLIND_FRAMES_DIR="${BLIND_FRAMES_DIR:-/data/yishuoyang/dataset/decase_challenge/frames_eval_track2_blind}"
FRAMES_BASE="${FRAMES_BASE:-${SHADOW}/frames_dev}"
MIC_BASE="${MIC_BASE:-${SHADOW}/foa_dev}"
DUMMY_LABELS="${DUMMY_LABELS:-${PKG}/blind/dummy_labels_dev}"
OUT_ROOT="${OUT_ROOT:-${PKG}/blind/v16_raw_track2_blind}"
MERGED="${MERGED:-${OUT_ROOT}/merged_preds}"
CKPT="${CKPT:-${PKG}/checkpoints/v16_energy_seg_epoch_001.pth}"
EXP_DIR_26="${EXP_DIR_26:-${ROOT}/experiments/formal_v16_clean_weakclass_from_fair_v11_20260610_1810_v16_clean_weak_small_g7_b1}"
GPUS_CSV="${GPUS_CSV:-0,1,2}"

export LD_LIBRARY_PATH="/home/yishuoyang/anaconda3/envs/audio_visual_seld_dcase2023/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

export DCASE2026_FRAMES_BASE="${FRAMES_BASE}"
export DCASE2026_LABELS_BASE="${DUMMY_LABELS}"
export DCASE2026_MIC_BASE="${MIC_BASE}"
export DCASE2026_UPLAM_CHECKPOINT="${DCASE2026_UPLAM_CHECKPOINT:-${ROOT}/UpLAM.pth}"
export DCASE2026_TEST_SPLIT_REGEX="${DCASE2026_BLIND_SPLIT_REGEX:-^dev-test-blind$}"
export DCASE2026_TEST_SPLIT_EXCLUDE_REGEX="${DCASE2026_BLIND_SPLIT_EXCLUDE_REGEX:-}"
export DCASE2026_SPLIT_INCLUDE_REGEX=""
export DCASE2026_SPLIT_EXCLUDE_REGEX=""

mkdir -p "${OUT_ROOT}" "${MERGED}" "${PKG}/logs"
if [[ ! -f "${PKG}/model_config.json" && -f "${PKG}/configs/v16_model_config.json" ]]; then
  cp -f "${PKG}/configs/v16_model_config.json" "${PKG}/model_config.json"
fi
if [[ ! -f "${PKG}/checkpoints/model_config.json" && -f "${PKG}/configs/v16_model_config.json" ]]; then
  cp -f "${PKG}/configs/v16_model_config.json" "${PKG}/checkpoints/model_config.json"
fi

"${PY}" "${PKG}/scripts/prepare_track2_blind_shadow.py" \
  --track2-root "${TRACK2_ROOT}" \
  --frames-dir "${BLIND_FRAMES_DIR}" \
  --shadow-root "${SHADOW}" \
  --split-name dev-test-blind

"${PY}" "${PKG}/scripts/prepare_blind_dummy_labels.py" \
  --metadata-dir "${SHADOW}/metadata_dev/dev-test-blind" \
  --out-labels-base "${DUMMY_LABELS}" \
  --split-name dev-test-blind

cd "${ROOT}"

SEQ_FILE="${OUT_ROOT}/blind_sequences.txt"
"${PY}" - <<'PY' > "${SEQ_FILE}" 2> "${OUT_ROOT}/sequence_discovery.log"
import os
import sys
from baseline_paths import FRAMES_BASE, LABELS_BASE
from model import get_sequence_infos

print(f"FRAMES_BASE={FRAMES_BASE}", file=sys.stderr)
print(f"LABELS_BASE={LABELS_BASE}", file=sys.stderr)
print(f"DCASE2026_TEST_SPLIT_REGEX={os.environ.get('DCASE2026_TEST_SPLIT_REGEX','')}", file=sys.stderr)
infos = get_sequence_infos("test", LABELS_BASE, FRAMES_BASE)
print(f"n_infos={len(infos)}", file=sys.stderr)
for _json, _seq_dir, seq_name in infos:
    print(seq_name)
PY

mapfile -t ALL_SEQS < "${SEQ_FILE}"
if [[ -n "${DCASE2026_BLIND_SEQS:-}" ]]; then
  IFS=',' read -r -a REQUESTED_SEQS <<< "${DCASE2026_BLIND_SEQS}"
  FILTERED=()
  for want in "${REQUESTED_SEQS[@]}"; do
    want="${want//[[:space:]]/}"
    [[ -z "${want}" ]] && continue
    found=0
    for seq in "${ALL_SEQS[@]}"; do
      if [[ "${seq}" == "${want}" ]]; then
        FILTERED+=("${seq}")
        found=1
        break
      fi
    done
    if [[ "${found}" -ne 1 ]]; then
      echo "[blind-v16] requested sequence not found: ${want}" >&2
      exit 3
    fi
  done
  ALL_SEQS=("${FILTERED[@]}")
  printf '%s\n' "${ALL_SEQS[@]}" > "${SEQ_FILE}"
fi
N_SEQ="${#ALL_SEQS[@]}"
IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
N_GPU="${#GPUS[@]}"

if [[ "${N_SEQ}" -lt 1 ]]; then
  echo "[blind-v16] no blind sequences discovered" >&2
  exit 2
fi

cat > "${OUT_ROOT}/run_metadata.txt" <<META
checkpoint=${CKPT}
exp_dir=${EXP_DIR_26}
frames_base=${FRAMES_BASE}
mic_base=${MIC_BASE}
dummy_labels=${DUMMY_LABELS}
gpus_csv=${GPUS_CSV}
num_sequences=${N_SEQ}
score_thr=${DCASE2026_BLIND_SCORE_THR:-0.22}
nms_iou_thr=${DCASE2026_BLIND_NMS_IOU_THR:-0.30}
max_dets_per_class=${DCASE2026_BLIND_MAX_DETS_PER_CLASS:-5}
min_hits=${DCASE2026_BLIND_MIN_HITS:-2}
energy_top_k=${DCASE2026_BLIND_ENERGY_TOP_K:-100}
energy_export_mode=${DCASE2026_BLIND_ENERGY_EXPORT_MODE:-hull}
energy_export_thr=${DCASE2026_BLIND_ENERGY_EXPORT_THR:-0.10}
energy_region_rel_thr=${DCASE2026_BLIND_REGION_REL_THR:-0.20}
energy_region_pad=${DCASE2026_BLIND_REGION_PAD:-1}
max_json_mb=${DCASE2026_BLIND_MAX_JSON_MB:-19.5}
META

echo "[blind-v16] START $(date '+%F %T') sequences=${N_SEQ} gpus=${GPUS_CSV}"

PIDS=()
for shard_idx in "${!GPUS[@]}"; do
  gpu="${GPUS[$shard_idx]}"
  shard_dir="${OUT_ROOT}/shard${shard_idx}_gpu${gpu}"
  pred_dir="${shard_dir}/preds"
  seqs_path="${shard_dir}/seqs.txt"
  mkdir -p "${pred_dir}"
  : > "${seqs_path}"

  for seq_idx in "${!ALL_SEQS[@]}"; do
    if (( seq_idx % N_GPU == shard_idx )); then
      printf '%s\n' "${ALL_SEQS[$seq_idx]}" >> "${seqs_path}"
    fi
  done
  seqs_csv="$(paste -sd, "${seqs_path}")"
  seq_count="$(wc -l < "${seqs_path}")"
  echo "[blind-v16] shard=${shard_idx} gpu=${gpu} seqs=${seq_count}"

  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${PY}" run_inference.py \
      --exp_dir "${EXP_DIR_26}" \
      --checkpoint "${CKPT}" \
      --out_dir "${pred_dir}" \
      --split test \
      --seqs "${seqs_csv}" \
      --batch_size "${DCASE2026_BLIND_BATCH_SIZE:-1}" \
      --num_workers "${DCASE2026_BLIND_NUM_WORKERS:-0}" \
      --score_thr "${DCASE2026_BLIND_SCORE_THR:-0.22}" \
      --nms_iou_thr "${DCASE2026_BLIND_NMS_IOU_THR:-0.30}" \
      --max_dets_per_class "${DCASE2026_BLIND_MAX_DETS_PER_CLASS:-5}" \
      --min_hits "${DCASE2026_BLIND_MIN_HITS:-2}" \
      --track_mean_score_thr 0.0 \
      --track_peak_score_thr 0.0 \
      --energy_top_k "${DCASE2026_BLIND_ENERGY_TOP_K:-100}" \
      --energy_export_thr "${DCASE2026_BLIND_ENERGY_EXPORT_THR:-0.10}" \
      --energy_source "${DCASE2026_BLIND_ENERGY_SOURCE:-roi}" \
      --energy_export_mode "${DCASE2026_BLIND_ENERGY_EXPORT_MODE:-hull}" \
      --energy_region_rel_thr "${DCASE2026_BLIND_REGION_REL_THR:-0.20}" \
      --energy_region_pad "${DCASE2026_BLIND_REGION_PAD:-1}" \
      --max_json_mb "${DCASE2026_BLIND_MAX_JSON_MB:-19.5}" \
      --iou_thr "${DCASE2026_BLIND_TRACK_IOU_THR:-0.30}" \
      --max_age "${DCASE2026_BLIND_MAX_AGE:-5}" \
      --coast_decay "${DCASE2026_BLIND_COAST_DECAY:-0.90}" \
      --gpu_acoustic \
      > "${shard_dir}/inference.log" \
      2> "${shard_dir}/inference_error.log"
  ) &
  PIDS+=("$!")
done

failed=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "[blind-v16] at least one shard failed; inspect ${OUT_ROOT}/shard*_gpu*/inference_error.log" >&2
  exit 4
fi

rm -rf "${MERGED}"
mkdir -p "${MERGED}"
for shard_idx in "${!GPUS[@]}"; do
  gpu="${GPUS[$shard_idx]}"
  cp "${OUT_ROOT}/shard${shard_idx}_gpu${gpu}/preds/"*_inference.json "${MERGED}/"
done

pred_count="$(find "${MERGED}" -maxdepth 1 -type f -name '*_inference.json' | wc -l)"
echo "[blind-v16] merged_predictions=${pred_count}/${N_SEQ}"
if [[ "${pred_count}" -ne "${N_SEQ}" ]]; then
  exit 5
fi

echo "[blind-v16] DONE $(date '+%F %T') merged=${MERGED}"
