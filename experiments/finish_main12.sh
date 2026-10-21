#!/usr/bin/env bash
set -e
cd /media/yjz/htr/ProSIU-main
export PYTHONPATH=experiments
export HF_HUB_OFFLINE=1
logs=/media/yjz/htr/ProSIU/studies/followup_main12/logs
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m followup.run E12 > "$logs/E12.log" 2>&1
while kill -0 "$1" 2>/dev/null; do sleep 5; done
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u -m followup.audit > "$logs/audit.log" 2>&1
.venv/bin/python -u -m followup.export > "$logs/export.log" 2>&1
echo 'Completed E01-E12 and result audit' > "$logs/completed.txt"
