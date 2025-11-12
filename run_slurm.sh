#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --constraint=h100
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00:00

if [ "$#" -ne 1 ]; then
    echo Usage: run_slurm.sh RUN_ID
    exit
fi
run_id=$1

export OMP_NUM_THREADS=1

# Read dataset name from config.
config_path=./configs/${run_id}.json
dataset_name=$(python3 -c '
import json
with open("'$config_path'", "r") as f:
    config = json.load(f)
if isinstance(config, dict):
    print(config["dataset"])
elif isinstance(config, list):
    print(config[0]["dataset"])
else:
    raise ValueError("Unrecognized config format.")
')

# If using imagenet, set up dataset on local drive.
if [[ "$dataset_name" == "imagenet" ]]; then
    data_dir="/tmp/"
    dataset_dir="/tmp/imagenet"

    # Avoid race condition with other processes.
    lockfile="/tmp/process_imagenet.lock"
    exec 9>"$lockfile"
    flock 9

    if [[ ! -d "$dataset_dir" ]]; then
        cp -r /mnt/home/gkrawezik/ceph/AI_DATASETS/ImageNet/2012/imagenet $dataset_dir
        pushd $data_dir
        mkdir train val
        cd train/
        unzip -qq ../train.zip
        cd ../val/
        unzip -qq ../val.zip
        popd
    fi

    # Release lock.
    flock -u 9
else
    data_dir="/mnt/ceph/users/mcrawshaw/stepback_data"
fi

module load python
source stepback_env/bin/activate
python3 run.py -i $run_id -nw 8 --data-dir $data_dir
