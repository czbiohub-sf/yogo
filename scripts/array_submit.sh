#!/bin/bash

#SBATCH --job-name=yogoUgandaInfer
#SBATCH --output=slurm-outputs/array/%A_%a.out
#SBATCH --error=slurm-outputs/array/%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=1-463%8
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1

# check if there is an input
if [ -z "$2" ]; then
  echo "usage: $0 <path to yogo pth> <path-to-file>"
  exit 1
fi

PTH_FILE="$1"
PARENT_PATH=$(dirname "$1")
MODEL_NAME=$(basename "$PARENT_PATH")

ZARR_PARENT_DIR_PATH=$(sed -n "$SLURM_ARRAY_TASK_ID"p "$2")
IMAGES_PARENT_DIR_PATH=$(echo "$ZARR_PARENT_DIR_PATH" | sed -E 's/Uganda_full(_[23])?/&_images/g')
RUN_NAME=$(basename "$IMAGES_PARENT_DIR_PATH")

NPY_OUTPUT_DIR="${IMAGES_PARENT_DIR_PATH}/yogo_preds_npy/$MODEL_NAME"

MASK_PATH="/hpc/projects/group.bioengineering/LFM_scope/Uganda_heatmaps/thresh_90/masks_npy"

if [ ! -d "$IMAGES_PARENT_DIR_PATH/images" ]; then
   >&2 echo "$IMAGES_PARENT_DIR_PATH/images doesn't exist"
  exit 1
fi

if [ ! -d "$ZARR_PARENT_DIR_PATH/sub_sample_imgs" ]; then
   >&2 echo "$ZARR_PARENT_DIR_PATH/sub_sample_imgs doesn't exist"
  exit 1
fi

mkdir -p "$NPY_OUTPUT_DIR"

out=$(
  conda run yogo infer \
    "$PTH_FILE" \
    --path-to-images "${IMAGES_PARENT_DIR_PATH}/images" \
    --min-class-confidence-threshold 0.90 \
    --heatmap-mask-path "$MASK_PATH/$RUN_NAME.npy" \
    --output-dir "$NPY_OUTPUT_DIR" \
    --save-npy
)

# if the prev command is successful, update which output is the latest"
if [ $? -eq 0 ]; then
  echo "$PTH_FILE" > "${IMAGES_PARENT_DIR_PATH}/yogo_preds_np/latest.txt"
else
  echo "Error occurred during inference on $IMAGES_PARENT_DIR_PATH" >&2
  echo "$out" >&2
fi
