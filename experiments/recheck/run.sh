#!/usr/bin/env bash
set -e
cd /media/yjz/htr/ProSIU-main
export PYTHONPATH=experiments
export HF_HUB_OFFLINE=1
out=/media/yjz/htr/ProSIU/studies/followup_recheck_20261019
mkdir -p "$out"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/recheck/run.py --seeds 0 1 > "$out/gpu0.log" 2>&1 &
p0=$!
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/recheck/run.py --seeds 2 3 4 > "$out/gpu1.log" 2>&1 &
p1=$!
wait "$p0"
wait "$p1"
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/recheck/verify.py > "$out/verification.log" 2>&1
