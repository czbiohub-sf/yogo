#! /bin/bash

#SBATCH --job-name=YOGOSweep
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=preempted
#SBATCH --array 1-256
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-task=32
#SBATCH --output=./%j.out

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
