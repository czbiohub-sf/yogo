# Where are sync points

Pytorch says [this](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#avoid-unnecessary-cpu-gpu-synchronization)

We find these:

```python3
yogo_loss.py:154                             centers_to_corners(pred_batch[i, :4, j, k])
cluster_anchors.py:107                       torch.minimum(b1[..., [1, 3]], b2[..., [1, 3]]) and b1[..., [0, 2]], b2[..., [0, 2]]
yogo_loss.py:158                             output[i, 0, j, k] = 1
train.py:79                                  {"train loss": loss.item(), "epoch": epoch},
dataloader.py:287                            return batched_inputs.to(device), [torch.tensor(l).to(device) for l in labels]
train.py:95                                  val_loss += loss.item()
utils.py:91                                  rects = [r for r in rects.reshape(pred_dim, Sx * Sy).T if r[4] > thresh]
utils.py:104                                 for r in rects
utils.py:23                                  if torch.all(img_labels[0, ...] == 0).item():
utils.py:48                                  "boxes": row_ordered_img_labels[mask, 1:5],
utils.py:49                                  "labels": row_ordered_img_labels[mask, 5],
utils.py:54                                  "boxes": row_ordered_img_preds[mask, :4],
utils.py:55                                  "scores": row_ordered_img_preds[mask, 4],
utils.py:56                                  "labels": torch.argmax(row_ordered_img_preds[mask, 5:], dim=1),
torchmetrics/detection/mean_ap.py:441        return torch.cat(self.detection_labels + self.groundtruth_labels).unique().tolist()
```

## Summary

1. indexing / slicing (`A[1,2,3:]`)
2. `.item()`
3. `to(device)`
4. loops?

# What can we fix?



- [A tensor can only be backed with a continuous chunk of memory, with a single stride per dim](https://discuss.pytorch.org/t/tensor-slice-views/24694/6) - this makes indexing an unfortunate operation, since it must be done on the CPU.
  - maybe we minimize the number
