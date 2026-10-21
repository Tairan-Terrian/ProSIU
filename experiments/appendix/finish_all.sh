#!/usr/bin/env bash
set -e
cd /media/yjz/htr/ProSIU-main
logs=/media/yjz/htr/ProSIU/studies/appendix_20261020/logs
while [ ! -f "$logs/gpu0_completed.txt" ] || [ ! -f "$logs/gpu1_completed.txt" ]; do sleep 30; done
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=experiments
export HF_HUB_OFFLINE=1
.venv/bin/python -u -m appendix.derive_panels
.venv/bin/python -u -m appendix.summarize --export-only
.venv/bin/python -u -m appendix.verify_results
printf 'All appendix experiments exported and independently verified\n' > "$logs/all_verified.txt"
