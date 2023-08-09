# Training

This document will discuss the training process that I (Axel) undertook to get YOGO up until now, along with any other notes that I could think of.

Note that there are also other *ways* to train. Just be careful to isolate your testing set from your training set, and to carefully look at the right metrics.

Finally, look at the [cli guide](https://github.com/czbiohub-sf/yogo/blob/main/docs/cli.md#yogo-train) for brief instructions on `yogo train`.

## Training: A Dialogue

*A* Ah yes, beauty divine! Labels assigned! YOGO shall know, though how does it learn?

*B* Quite simply, my good friend! `yogo train path_to/dataset_defn.yml` will teach that model a thing or two (or seven?).

*A* Wait, what is this `dataset_defn.yml`?

*B* Simply [this](dataset_description.md) - a way to define your dataset. In short, tells the dataloader how to split the data for testing, validation and training.

*A* We already have many datasets of labelled data, where are those definition files?

*B* Our dataset definition files and labels for the data are in `biohub_labels` (full path `/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/biohub-labels`). The runs that I commonly 

*A* Ah, makes sense. My data is on Flexo, so I'll just mount it to my laptop and train!

*B* Patience is sometimes not a virtue. That will take too long, `ssh` into Bruno and borrow a GPU. The scripts directory has some of what you need; `submit_cmd.sh` requests one GPU, `submit_cmd_multi_gpu.sh` requests four, and `array_submit.sh` can requests thousands.

*A* Good god! OK, I guess `array_submit.sh` isn't typically used for training?

*B* Correct, do NOT use that for training.

*A* Roger. How do I use `submit_cmd.sh`?

*B* Assuming you're in the root YOGO directory, `sbatch scripts/submit_cmd(_multi_gpu.sh) yogo train path_to/dataset_defn.yml` will submit that job to Bruno. When there are resources, YOGO will start running the script.

*A* Alright, it has been run! How do I know when the job will start? When it will stop?

*B* The `squeue` command will print the entire "Slurm" queue. Slurm is the job management program that Bruno uses, and it is how you borrow GPUs, or any other computational resource on Bruno. Simply typing `squeue` will print the status of *everyone's* jobs. Good for checking how much Bruno is being used. But, to see only your jobs, `squeue --me` is good:

```console
[axel.jacobsen@login-02 yogo]$ squeue --me
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          10644479       gpu YOGOTrai axel.jac PD       0:00      1 (Priority)
```

This shows the Job ID, where you are running the job (`gpu` if on a gpu node), the runtime (which will show zero until it is running), and if the job is waiting, `(Priority)` or `(Resources)`. Once the job starts, it'll show something like `gpu-a-2`.

*A* Cool. Should I use `submit_cmd_multi_gpu.sh` or `submit_cmd.sh`?

*B* The multi-gpu script will run much faster, but sometimes Bruno is being battered with jobs. In which case, it'll be easier to just grab one with `submit_cmd.sh`.

*A* OK, cool! The model has been running for a bit now, how can I check on it's progress?

*B* Weights and Biases. You need a seat on our account. Assuming you have one, you should see it there.

*A* Oh

*B* 
