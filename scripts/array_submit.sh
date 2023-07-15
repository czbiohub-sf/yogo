#!/bin/bash

#SBATCH --job-name=yogoUgandaInfer
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=1-961%8
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=64

FILE_PATH=$(sed -n "$SLURM_ARRAY_TASK_ID"p uganda_runs.txt)
FILE_NAME=$(basename "$FILE_PATH")

echo $FILE_PATH > "results/${FILE_NAME}.txt"

conda run yogo infer \
  /home/axel.jacobsen/celldiagnosis/yogo/trained_models/fallen-wind-1668/best.pth \
  --path-to-images "${FILE_PATH}/images" \
  --count >> "results/${FILE_NAME}.txt"
