#!/bin/bash

#SBATCH --job-name=yogoUgandaInfer
#SBATCH --output=temp_output/logs/%A_%a.out
#SBATCH --error=temp_output/logs/%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=1-360%16
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1

# check if there is an input
if [ -z "$2" ]; then
  echo "usage: $0 <path to yogo pth> <path-to-file>"
  exit 1
fi

PTH_FILE="$1"

FILE_PATH=$(sed -n "$SLURM_ARRAY_TASK_ID"p "$2")
FILE_NAME=$(basename "$FILE_PATH")

MASK_PATH="/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/Uganda_heatmaps/masks_npy"

if [ ! -d "${FILE_PATH}/images" ]; then
   >&2 echo "${FILE_PATH}/images doesn't exist"
  exit 1
fi

if [ ! -d "${FILE_PATH//_images/}/sub_sample_imgs" ]; then
   >&2 echo "${FILE_PATH//_images/}/sub_sample_imgs doesn't exist"
  exit 1
fi

mkdir -p temp_output/results

out=$(
  conda run yogo infer \
    "$PTH_FILE" \
    --path-to-images "${FILE_PATH}/images" \
    --min-class-confidence-threshold 0.95 \
    --heatmap-mask-path "$MASK_PATH/$FILE_NAME.npy" \
    --count
)

# if the prev command is successful, pipe output to "results/${FILE_NAME}.txt"
if [ $? -eq 0 ]; then
  echo $FILE_PATH > "temp_output/results/${FILE_NAME}.txt"
  echo "$out" >> "temp_output/results/${FILE_NAME}.txt"
fi
