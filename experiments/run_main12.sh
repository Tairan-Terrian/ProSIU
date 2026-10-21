#!/usr/bin/env bash
set -e
cd /media/yjz/htr/ProSIU-main
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=experiments
export HF_HUB_OFFLINE=1
logs=/media/yjz/htr/ProSIU/studies/followup_main12/logs
for experiment in ${*:-E01 E02 E03 E04 E05 E06 E07 E08 E09 E10 E11 E12}; do
    .venv/bin/python -u -m followup.run "$experiment" > "$logs/$experiment.log" 2>&1
    if [ "$experiment" = E11 ]; then
        .venv/bin/python -u -m followup.native_e11 > "$logs/E11_native.log" 2>&1
    fi
done
.venv/bin/python -u -m followup.audit > "$logs/audit.log" 2>&1
.venv/bin/python -u -m followup.export > "$logs/export.log" 2>&1
echo 'Completed E01-E12 and result audit' > "$logs/completed.txt"
