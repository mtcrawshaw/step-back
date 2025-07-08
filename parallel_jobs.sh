#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo Usage: parallel_jobs.sh CONFIG_ID
    exit
fi
config_id=$1
config_dir="configs"
output_dir="output"

# Split large config into many configs with a single run each.
python <<EOF
from stepback.utils import split_config
split_config("$config_id", "$config_id", "$config_dir")
EOF

# Launch each single run as a separate slurm job.
num_configs=$(ls ${config_dir}/${config_id}/*.json | wc -l)
job_ids=()
for i in $(seq 0 $(($num_configs - 1)) | xargs -I{} printf "%02d\n" {}); do
    run_id=${config_id}/${config_id}-${i}
    log_file=${config_dir}/${run_id}.log
    results_file=${output_dir}/${run_id}.json

    if [ -e "$results_file" ]; then
        echo "Log $results_file already exists. Skipping this one."
    else
        job_id=$(sbatch -o $log_file run_slurm.sh $run_id | awk '{print $4}')
        echo "Submitted job ${job_id} for $log_file."
        job_ids+=($job_id)
    fi
done

# Wait for all jobs to finish.
while [ ${#job_ids[@]} -gt 0 ]; do

    still_running=()
    for id in "${job_ids[@]}"; do
        if squeue -j "$id" | grep -q "$id"; then
            still_running+=("$id")
        else
            echo "Job $id finished."
        fi
    done

    job_ids=("${still_running[@]}")
    if [ ${#job_ids[@]} -gt 0 ]; then
        sleep 5
    fi
done

# Merge output files.
python <<EOF
from stepback.utils import merge_subfolder
merge_subfolder("$config_id", "$config_id")
EOF
