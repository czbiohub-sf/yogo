#!/bin/bash

#SBATCH --job-name=ULCMalariaYOGOTraining
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=./slurm-outputs/slurm-%j.out

# TODO: See docs for potentially cleaner way to do this https://slurm.schedmd.com/sbatch.html

echo
echo "You Only Glance Once (YOGO) Sweep"
echo

env | grep "^SLURM" | sort

echo
echo "starting yogo training..."
echo

nvidia-smi

wandb online
conda run wandb agent --count 1 "$@"
wandb offline
