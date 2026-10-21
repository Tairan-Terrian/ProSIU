#!/usr/bin/env bash
set -e
cd /media/yjz/htr/ProSIU-main
export PYTHONPATH=experiments
export HF_HUB_OFFLINE=1
logs=/media/yjz/htr/ProSIU/studies/followup_main12/logs
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m followup.run E12 --seeds 0 2 4 > "$logs/E12_even.log" 2>&1 &
even_pid=$!
while kill -0 "$1" 2>/dev/null; do sleep 5; done
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u -m followup.run E12 --seeds 1 3 > "$logs/E12_odd.log" 2>&1 &
odd_pid=$!
wait "$even_pid"
wait "$odd_pid"
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u -m followup.audit > "$logs/audit.log" 2>&1
.venv/bin/python -u -m followup.export > "$logs/export.log" 2>&1
echo 'Completed E01-E12 and result audit' > "$logs/completed.txt"
