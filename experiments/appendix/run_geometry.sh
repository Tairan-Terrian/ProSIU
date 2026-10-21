#!/usr/bin/env bash
set -e
cd /media/yjz/htr/ProSIU-main
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=experiments
export HF_HUB_OFFLINE=1
.venv/bin/python -u -m appendix.diagnostics --experiments theory E15 E41 --seeds 0 1 2 3 4
