#!/usr/bin/env bash
set -e
cd /media/yjz/htr/ProSIU-main
export PYTHONPATH=experiments
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1
.venv/bin/python experiments/revision/ablation.py
.venv/bin/python experiments/revision/reconstruction.py
