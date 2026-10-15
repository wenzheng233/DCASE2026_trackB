#!/usr/bin/env bash
set -o pipefail
PKG=/home/yishuoyang/AV-SELD/v16_23prior \
GPUS_CSV=1,2,5 \
DCASE2026_BLIND_BATCH_SIZE=1 \
bash /home/yishuoyang/AV-SELD/v16_23prior/scripts/run_v16_track2_blind_inference.sh \
  2>&1 | tee /home/yishuoyang/AV-SELD/v16_23prior/logs/run_v16_track2_blind_inference_tmux.log
status=${PIPESTATUS[0]}
echo "${status}" > /home/yishuoyang/AV-SELD/v16_23prior/logs/run_v16_track2_blind_inference.exit
exit "${status}"
