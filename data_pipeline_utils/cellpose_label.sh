#!/bin/bash

#SBATCH --job-name=Cellpose_Labelling
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=16
#SBATCH --output=./slurm-%j.out

env | grep "^SLURM" | sort

conda run python3 generate_cellpose_labels.py "$@"
