#! /usr/bin/env bash

sbatch scripts/submit_cmd_multi_gpu_light.sh yogo train \
   ../dataset_defs/pre-training/yogo_parasite_data_with_tests.yml \
   --lr 5e-4 \
   --batch-size 64 \
   --epochs 32 \
   --label-smoothing 1e-2 \
   --weight-decay 5e-2 \
   --no-obj-weight 1 \
   --note "depth_ver_0" \
   --lr-decay-factor 16 \
   --model depth_ver_0 \
   --tags scaling-law depthwise \
   --normalize-images

sbatch scripts/submit_cmd_multi_gpu_light.sh yogo train \
   ../dataset_defs/pre-training/yogo_parasite_data_with_tests.yml \
   --lr 5e-4 \
   --batch-size 64 \
   --epochs 32 \
   --label-smoothing 1e-2 \
   --weight-decay 5e-2 \
   --no-obj-weight 1 \
   --note "depth_ver_1" \
   --lr-decay-factor 16 \
   --model depth_ver_1 \
   --tags scaling-law depthwise \
   --normalize-images

sbatch scripts/submit_cmd_multi_gpu_light.sh yogo train \
   ../dataset_defs/pre-training/yogo_parasite_data_with_tests.yml \
   --lr 5e-4 \
   --batch-size 64 \
   --epochs 32 \
   --label-smoothing 1e-2 \
   --weight-decay 5e-2 \
   --no-obj-weight 1 \
   --note "depth_ver_2" \
   --lr-decay-factor 16 \
   --model depth_ver_2 \
   --tags scaling-law depthwise \
   --normalize-images

sbatch scripts/submit_cmd_multi_gpu.sh yogo train \
   ../dataset_defs/pre-training/yogo_parasite_data_with_tests.yml \
   --lr 5e-4 \
   --batch-size 64 \
   --epochs 32 \
   --label-smoothing 1e-2 \
   --weight-decay 5e-2 \
   --no-obj-weight 1 \
   --note "depth_ver_3" \
   --lr-decay-factor 16 \
   --model depth_ver_3 \
   --tags scaling-law depthwise \
   --normalize-images

sbatch scripts/submit_cmd_multi_gpu.sh yogo train \
   ../dataset_defs/pre-training/yogo_parasite_data_with_tests.yml \
   --lr 5e-4 \
   --batch-size 64 \
   --epochs 32 \
   --label-smoothing 1e-2 \
   --weight-decay 5e-2 \
   --no-obj-weight 1 \
   --note "depth_ver_4" \
   --lr-decay-factor 16 \
   --model depth_ver_4 \
   --tags scaling-law depthwise \
   --normalize-images
