import torch

from copy import deepcopy

from yogo.model import YOGO


def checkpoint(filepath, net, epoch, global_step):
    "simplified, from yogo.train.Trainer.checkpoint"
    torch.save(
        {
            "epoch": epoch,
            "step": global_step,
            "model_state_dict": deepcopy(net.state_dict()),
            "model_version": net.model.__name__,
        },
        str(filepath),
    )


def test_model_io_basic(tmpdir):
    "shouldn't throw an error"
    y = YOGO(img_size=(772, 1032), anchor_w=0.05, anchor_h=0.05, num_classes=7)
    checkpoint(tmpdir / "test.pth", y, 0, 0)
    z = YOGO.from_pth(tmpdir / "test.pth")
    assert y.img_size == z.img_size
    assert y.anchor_w == z.anchor_w
    assert y.anchor_h == z.anchor_h
    assert y.num_classes == z.num_classes
    assert y.is_rgb == z.is_rgb
    assert y.normalize_images == z.normalize_images
    assert y.clip_value == z.clip_value
    assert y.height_multiplier == z.height_multiplier
    assert y.width_multiplier == z.width_multiplier
    assert y.model == z.model
