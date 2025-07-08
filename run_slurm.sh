#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --constraint=h100
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

if [ "$#" -ne 1 ]; then
    echo Usage: run_slurm.sh RUN_ID
    exit
fi
run_id=$1

export OMP_NUM_THREADS=1
data_dir="/mnt/ceph/users/mcrawshaw/stepback_data"

module load python
source stepback_env/bin/activate
python3 run.py -i $run_id -nw 8 --data-dir $data_dir
