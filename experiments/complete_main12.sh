#!/usr/bin/env bash
set -e
cd /media/yjz/htr/ProSIU-main
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=experiments
export HF_HUB_OFFLINE=1
logs=/media/yjz/htr/ProSIU/studies/followup_main12/logs
for experiment in E01 E09 E10; do
    .venv/bin/python -u -m followup.run "$experiment" --methods SCRUB --replace > "$logs/${experiment}_SCRUB.log" 2>&1
done
bash experiments/run_main12.sh E10 E11 E12
