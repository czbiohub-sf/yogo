#!/bin/bash

#SBATCH --job-name=YOGOProfiling
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1

echo
echo "You Only Glance Once (YOGO) Profiling"
echo

env | grep "^SLURM" | sort

echo
echo "starting yogo profiling..."
echo

nvidia-smi

conda run python3 train.py "$@"
