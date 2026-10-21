#!/usr/bin/env bash
set -e
cd /media/yjz/htr/ProSIU-main
export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH=experiments
export HF_HUB_OFFLINE=1
.venv/bin/python -u -m appendix.derive_panels
.venv/bin/python -u -m appendix.representations --seeds 0 1 2 3 4
.venv/bin/python -u -m appendix.agents --experiments E36 --seeds 0 1 2 3 4
