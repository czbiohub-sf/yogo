# Training

This document will cover the practical details of training YOGO - where to look for data, how to format your data, how to get a GPU, e.t.c.

Finally, look at the [cli guide](https://github.com/czbiohub-sf/yogo/blob/main/docs/cli.md#yogo-train) for brief instructions on `yogo train`.

## Dataset Definition

The `dataset_defn.yml` file provides data definitions, guiding the dataloader on splitting data for testing, validation, and training. A detailed explanation is in [dataset-definition.md](dataset-definition.md). The dataset definition files are in [this repo](https://github.com/czbiohub-sf/lfm-dataset-definitions?tab=readme-ov-file), which is in this location on bruno: ` .../LFM_scope/biohub-labels/dataset_defs`.

Training is split into pre-training and fine-tuning. Pre-training is first completed on YOGO-labelled data, and fine-tuning is completed on human-labelled data.

Here are dataset definition files:

- **Pre-training:** `pre-training/yogo_parasite_data_with_tests.yml.yml` : Contains YOGO and hand-labelled test data. Most of the time, use existing pretrained models like "expert-night-1797" or "rare-valley-1798".

- **Fine-tuning:** `fine-tuning/all-dataset-subsets.yml` : Only hand-labelled data.

The last two files are often used for training. Add new data to them as needed.

## Accessing and Using GPUs

For efficient training, use a GPU by connecting to Bruno.

In the `yogo/scripts` directory:

- `submit_cmd.sh`: Asks for one GPU.
- `submit_cmd_multi_gpu.sh`: Asks for four GPUs.
- `array_submit.sh`: Can ask for many GPUs.

The multi-gpu scripts will train yogo much faster, but if Bruno is being heavily used, it may be difficult to get 4 gpus at once.

**Note:** Don't use `array_submit.sh` for training, it is intended for many short jobs that each require a gpu.

### Submitting Training Job

From YOGO's main directory, run:

```
sbatch scripts/submit_cmd(_multi_gpu.sh) yogo train path_to/dataset_defn.yml
```

with whatever hyperparameters / options that you want. Read `yogo train --help` and the [cli guide](https://github.com/czbiohub-sf/yogo/blob/main/docs/cli.md#yogo-train) for details

### Monitoring Job Status

Slurm manages jobs on Bruno. To check job statuses:

```console
squeue
squeue --me  // for your jobs
```
