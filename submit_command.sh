#!/bin/bash

#SBATCH --job-name=ULCMalariaYOGOTraining
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1

env | grep "^SLURM" | sort

nvidia-smi

wandb online
conda run "$@"
wandb offline

