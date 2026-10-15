#!/usr/bin/env bash
set -euo pipefail
bash /home/yishuoyang/AV-SELD/v16_23prior/scripts/run_v16_track2_blind_tail_supp_g34.sh 2>&1 | tee /home/yishuoyang/AV-SELD/v16_23prior/logs/run_v16_track2_blind_tail_supp_g34_tmux.log
status=${PIPESTATUS[0]}
echo ${status} > /home/yishuoyang/AV-SELD/v16_23prior/logs/run_v16_track2_blind_tail_supp_g34.exit
exit ${status}
