#! /usr/bin/env bash

sbatch scripts/submit_cmd_multi_gpu.sh yogo train \
   ../dataset_defs/human-labels/all-dataset-subsets-no-aug.yml \
   --from-pretrained trained_models/rural-dragon-580/best.pth \
   --lr 5e-5 \
   --batch-size 64 \
   --epochs 128 \
   --label-smoothing 5e-3 \
   --weight-decay 5e-3 \
   --no-obj-weight 1 \
   --note "depth_ver_0" \
   --lr-decay-factor 16 \
   --model depth_ver_0 \
   --tags scaling-law depthwise fine-tune \
   --normalize-images

sbatch scripts/submit_cmd_multi_gpu.sh yogo train \
   ../dataset_defs/human-labels/all-dataset-subsets-no-aug.yml \
   --from-pretrained trained_models/sweet-tree-579/best.pth \
   --lr 5e-5 \
   --batch-size 64 \
   --epochs 128 \
   --label-smoothing 5e-3 \
   --weight-decay 5e-3 \
   --no-obj-weight 1 \
   --note "depth_ver_1" \
   --lr-decay-factor 16 \
   --model depth_ver_1 \
   --tags scaling-law depthwise fine-tune \
   --normalize-images

sbatch scripts/submit_cmd_multi_gpu.sh yogo train \
   ../dataset_defs/human-labels/all-dataset-subsets-no-aug.yml \
   --from-pretrained trained_models/woven-night-577/best.pth \
   --lr 5e-5 \
   --batch-size 64 \
   --epochs 128 \
   --label-smoothing 5e-3 \
   --weight-decay 5e-3 \
   --no-obj-weight 1 \
   --note "depth_ver_2" \
   --lr-decay-factor 16 \
   --model depth_ver_2 \
   --tags scaling-law depthwise fine-tune \
   --normalize-images

sbatch --dependency=afterok:12365763 scripts/submit_cmd_multi_gpu.sh yogo train \
   ../dataset_defs/human-labels/all-dataset-subsets-no-aug.yml \
   --from-pretrained trained_models/helpful-frog-583/best.pth \
   --lr 5e-5 \
   --batch-size 64 \
   --epochs 128 \
   --label-smoothing 5e-3 \
   --weight-decay 5e-3 \
   --no-obj-weight 1 \
   --note "depth_ver_3" \
   --lr-decay-factor 16 \
   --model depth_ver_3 \
   --tags scaling-law depthwise fine-tune \
   --normalize-images

sbatch --dependency=afterok:12365764 scripts/submit_cmd_multi_gpu.sh yogo train \
   ../dataset_defs/human-labels/all-dataset-subsets-no-aug.yml \
   --from-pretrained trained_models/expert-cloud-582/best.pth \
   --lr 5e-5 \
   --batch-size 32 \
   --epochs 128 \
   --label-smoothing 5e-3 \
   --weight-decay 5e-3 \
   --no-obj-weight 1 \
   --note "depth_ver_4" \
   --lr-decay-factor 16 \
   --model depth_ver_4 \
   --tags scaling-law depthwise fine-tune \
   --normalize-images
