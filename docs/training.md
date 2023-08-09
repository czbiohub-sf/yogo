# Training

This document will discuss the training process that I (Axel) undertook to get YOGO to where it is now, along with any other notes that I could think of.

Note that there are also other *ways* to train. Just be careful to isolate your testing set from your training set, and to look at the right metrics.

Finally, look at the [cli guide](https://github.com/czbiohub-sf/yogo/blob/main/docs/cli.md#yogo-train) for brief instructions on `yogo train`.

## Dataset Definition

The `dataset_defn.yml` file provides data definitions, guiding the dataloader on splitting data for testing, validation, and training. A detailed explanation is in [dataset_definition.md](dataset_definition.md).

Dataset files and labels are in:

```
/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/biohub-labels
```

Common files include:

- **Pre-training:**
    - `pre-training/yogo_parasite_data_with_tests.yml` : Contains YOGO and hand-labelled test data. Most of the time, use existing pretrained models like "expert-night-1797" or "rare-valley-1798".

- **Human Labels:**
    - `human-labels/all-labelled-data-test.yml` : Only hand-labelled data.
    - `human-labels/all-labelled-data-test-good-healthy.yml` : Hand-labelled data and healthy YOGO labels. Parasite classes are marked as healthy.

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

# Training: A Dialogue

*A* Ah yes, beauty divine! Labels assigned! YOGO shall know, though how does it learn?

*B* Quite simply, my good friend! `yogo train path_to/dataset_defn.yml` will teach that model a thing or two (or seven?).

*A* Wait, what is this `dataset_defn.yml`?

*B* Simply [this](dataset_definition.md) - a way to define your dataset. In short, tells the dataloader how to split the data for testing, validation and training.

*A* We already have many datasets of labelled data, where are those definition files?

*B* Our dataset definition files and labels for the data are in `biohub_labels` (full path `/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/biohub-labels`). There are a couple scripts that I use often:

- `pre-training/yogo_parasite_data_with_tests.yml` is a file for "pre-training" - it is composed of YOGO labels for all of our data, and our hand-labelled sets for test data. 99% of the time, you won't have to train a new network, just continue training from one of our pretrained models, "expert-night-1797" and "rare-valley-1798".
- `human-labels/all-labelled-data-test.yml` consists of our hand-labelled data
- `human-labels/all-labelled-data-test-good-healthy.yml` consists of our hand-labelled data, plus some healthy runs with YOGO labels, with all parasite classes corrected to healthy.

The final two files are the files that should be used for training the most, and new labelled data should be added to that set. Ask about adding or updating data later.

*A* Ah, makes sense. The data is on Flexo, so I'll just mount it to my laptop and train!

*B* Patience is sometimes not a virtue. That will take too long, `ssh` into Bruno and borrow a GPU. The `scripts` directory has some of what you need; `submit_cmd.sh` requests one GPU, `submit_cmd_multi_gpu.sh` requests four, and `array_submit.sh` can requests thousands.

*A* Good god! OK, I guess `array_submit.sh` isn't typically used for training?

*B* Correct, do NOT use that for training.

*A* Roger. How do I use `submit_cmd_multi_gpu.sh`?

*B* Assuming you're in the root YOGO directory, `sbatch scripts/submit_cmd(_multi_gpu.sh) yogo train path_to/dataset_defn.yml` will submit that job to Bruno. When there are resources, YOGO will start running the script.

*A* Alright, it has been run! How do I know when the job will start? When it will stop?

*B* The `squeue` command will print the entire "Slurm" queue. Slurm is the job management program that Bruno uses, and it is how you borrow GPUs, or any other computational resource on Bruno. Simply typing `squeue` will print the status of *everyone's* jobs. Good for checking how much Bruno is being used. But, to see only your jobs, `squeue --me` is good:

```console
[axel.jacobsen@login-02 yogo]$ squeue --me
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          10644479       gpu YOGOTrai axel.jac PD       0:00      1 (Priority)
```

This shows the Job ID, where you are running the job (`gpu` if on a gpu node), the runtime (which will show zero until it is running), and if the job is waiting, `(Priority)` or `(Resources)`. Once the job starts, it'll show something like `gpu-a-2`.

*A* Should I use `submit_cmd_multi_gpu.sh` or `submit_cmd.sh`?

*B* The multi-gpu script will run much faster, but sometimes Bruno is being battered with jobs. In which case, it'll be easier to just grab one with `submit_cmd.sh`.

*A* OK, cool! The model has been running for a bit now, how can I check on it's progress?

*B* Weights and Biases. You need a seat on our account. Assuming you have one, you should see it there.

*A* Oh
