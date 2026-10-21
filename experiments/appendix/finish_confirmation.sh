#!/usr/bin/env bash
set -e
cd /media/yjz/htr/ProSIU-main
while kill -0 3698665 2>/dev/null || kill -0 3698666 2>/dev/null; do sleep 30; done
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=experiments
export HF_HUB_OFFLINE=1
.venv/bin/python -u -c 'import torch; torch.set_default_dtype(torch.float64); torch.set_num_threads(4); from appendix.summarize import confirmation_statistics, cases; confirmation_statistics(); cases()'
printf 'Completed independent-request statistics and request-stratified examples\n'
