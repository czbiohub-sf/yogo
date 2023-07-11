#! /bin/bash

#SBATCH --job-name=YOGOTraining
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1:a100
#SBATCH --cpus-per-task=64
#SBATCH --output=./slurm-outputs/%j.out

env | grep "^SLURM" | sort

nvcc --version

nvidia-smi

echo "running: $@"

wandb enabled
wandb online
conda run "$@"
wandb offline
wandb disabled
