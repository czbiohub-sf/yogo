#! /usr/bin/env bash

sbatch scripts/submit_cmd_multi_gpu_light.sh yogo train \
   ../dataset_defs/pre-training/yogo_parasite_data_with_tests.yml \
   --lr 5e-4 \
   --batch-size 64 \
   --epochs 32 \
   --label-smoothing 1e-2 \
   --weight-decay 5e-2 \
   --no-obj-weight 1 \
   --note "quarter_filters" \
   --lr-decay-factor 16 \
   --model quarter_filters \
   --tag scaling-law \
   --normalize-images

sbatch scripts/submit_cmd_multi_gpu_light.sh yogo train \
   ../dataset_defs/pre-training/yogo_parasite_data_with_tests.yml \
   --lr 5e-4 \
   --batch-size 64 \
   --epochs 32 \
   --label-smoothing 1e-2 \
   --weight-decay 5e-2 \
   --no-obj-weight 1 \
   --note "half_filters" \
   --lr-decay-factor 16 \
   --model half_filters \
   --tag scaling-law \
   --normalize-images

sbatch scripts/submit_cmd_multi_gpu_light.sh yogo train \
   ../dataset_defs/pre-training/yogo_parasite_data_with_tests.yml \
   --lr 5e-4 \
   --batch-size 64 \
   --epochs 32 \
   --label-smoothing 1e-2 \
   --weight-decay 5e-2 \
   --no-obj-weight 1 \
   --note "base" \
   --lr-decay-factor 16 \
   --model base_model \
   --tag scaling-law \
   --normalize-images

sbatch scripts/submit_cmd_multi_gpu_light.sh yogo train \
   ../dataset_defs/pre-training/yogo_parasite_data_with_tests.yml \
   --lr 5e-4 \
   --batch-size 64 \
   --epochs 32 \
   --label-smoothing 1e-2 \
   --weight-decay 5e-2 \
   --no-obj-weight 1 \
   --note "double_filters" \
   --lr-decay-factor 16 \
   --model double_filters \
   --tag scaling-law \
   --normalize-images

sbatch scripts/submit_cmd_multi_gpu_light.sh yogo train \
   ../dataset_defs/pre-training/yogo_parasite_data_with_tests.yml \
   --lr 5e-4 \
   --batch-size 64 \
   --epochs 32 \
   --label-smoothing 1e-2 \
   --weight-decay 5e-2 \
   --no-obj-weight 1 \
   --note "triple_filters" \
   --lr-decay-factor 16 \
   --model triple_filters \
   --tag scaling-law \
   --normalize-images
