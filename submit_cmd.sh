#!/bin/bash

#SBATCH --job-name=ULCMalariaYOGOTraining
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=./slurm-outputs/slurm-%j.out

env | grep "^SLURM" | sort

# curious about transfer time

# TODO this is bad :(
echo $(date '+%d/%m/%Y %H:%M:%S')
mkdir -p /tmp
tar -xf /hpc/projects/flexo/MicroscopyData/Bioengineering/LFM\ Scope/cellpose_data/training_data_sample_2022_11_01_cyto2.tar.gz -C /tmp/
echo $(date '+%d/%m/%Y %H:%M:%S')


nvidia-smi

wandb enabled
wandb online
conda run "$@"
wandb offline
wandb disabled
