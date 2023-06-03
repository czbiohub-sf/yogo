import torch
import wandb

from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from torch.optim import AdamW

from yogo.model import YOGO
from yogo.utils import draw_yogo_prediction
from yogo.yogo_loss import YOGOLoss
from yogo.data.blobgen import BlobDataset
from yogo.data.dataset import YOGO_CLASS_ORDERING


def collate_batch(batch):
    inputs, labels = zip(*batch)
    batched_inputs = torch.stack(inputs)
    batched_labels = torch.stack(labels)
    return batched_inputs, batched_labels


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_paths = {
    "healthy": "/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/training-data-thumbnails/healthy",
    "ring": "/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/training-data-thumbnails/ring",
    "schizont": "/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/training-data-thumbnails/schizont",
    "trophozoite": "/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/training-data-thumbnails/trophozoite",
    "gametocyte": "/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/training-data-thumbnails/gametocyte",
    "wbc": "/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/training-data-thumbnails/wbc",
    "misc": "/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/training-data-thumbnails/misc",
}

dl: DataLoader[ConcatDataset[BlobDataset]] = DataLoader(
    ConcatDataset(
        [
            BlobDataset(
                {k: v},
                129,
                97,
                n=12,
                length=800,
                blend_thumbnails=True,
            )
            for k, v in data_paths.items()
        ]
    ),
    batch_size=64,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
    collate_fn=collate_batch,
)


net = YOGO(
    num_classes=7,
    img_size=(772, 1032),
    anchor_w=0.4,
    anchor_h=0.5,
).to(device)

optimizer = AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)

wandb.init(
    project="yogo",
    entity="bioengineering",
    tags=["testing"],
)


Y_loss = YOGOLoss(
    label_smoothing=0.01,
    classify=True,
).to(device)

global_step = 0
for epoch in range(1000):
    for imgs, labels in dl:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        out = net(imgs)
        loss, loss_components = Y_loss(out, labels)

        loss.backward()

        optimizer.step()

        global_step += 1
        wandb.log(
            {
                "train loss": loss.item(),
                "epoch": epoch,
                "training grad norm": net.grad_norm(),
                "training param norm": net.param_norm(),
                **loss_components,
            },
            step=global_step,
        )

    annotated_img = wandb.Image(
        draw_yogo_prediction(
            imgs[0, 0, ...].cpu().int(),
            out[0, ...].cpu().detach(),
            thresh=0.5,
            labels=YOGO_CLASS_ORDERING,
        )
    )
    wandb.log({"validation bbs": annotated_img}, step=global_step)
