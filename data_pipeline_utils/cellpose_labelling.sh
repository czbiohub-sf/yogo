#! /usr/bin/env bash


# This will run cellpose on a directory of images.
# usage: ./cellpose_labelling.sh <your directory>
# best done via SLURM

(set -x;
  cellpose --dir "$@" --chan 0 --verbose --use_gpu --diameter=37 --batch_size 16 --pretrained_model "cyto2" --save_outlines;
  python3 clean_npy_results.py "$@")

