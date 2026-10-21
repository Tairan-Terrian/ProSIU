#!/usr/bin/env bash
set -e
cd /media/yjz/htr/ProSIU-main
gpu=$1
export CUDA_VISIBLE_DEVICES=$gpu
export PYTHONPATH=experiments
export HF_HUB_OFFLINE=1
logs=/media/yjz/htr/ProSIU/studies/appendix_20261020/logs
if [ "$gpu" = 0 ]; then
    seeds="0 2"
    extra_assays="amacGFP"
else
    seeds="1 3"
    extra_assays="cgreGFP avGFP"
fi
.venv/bin/python -u -m appendix.empirical --seeds $seeds >> "$logs/empirical_gpu$gpu.log" 2>&1
.venv/bin/python -u -m appendix.empirical --seeds 4 --assays $extra_assays >> "$logs/empirical_gpu$gpu.log" 2>&1
.venv/bin/python -u -m appendix.diagnostics --seeds $seeds >> "$logs/diagnostics_gpu$gpu.log" 2>&1
.venv/bin/python -u -m appendix.diagnostics --seeds 4 --assays $extra_assays >> "$logs/diagnostics_gpu$gpu.log" 2>&1
if [ "$gpu" = 1 ]; then
    .venv/bin/python -u -m appendix.representations --seeds 0 1 2 3 4 > "$logs/representations.log" 2>&1
    .venv/bin/python -u -m appendix.agents --experiments E36 --seeds 0 1 2 3 4 > "$logs/generation.log" 2>&1
else
    .venv/bin/python -u -m appendix.agents --experiments E35 E37 --seeds 0 1 2 3 4 > "$logs/agents.log" 2>&1
fi
printf 'Completed assigned appendix experiments\n' > "$logs/gpu${gpu}_completed.txt"
