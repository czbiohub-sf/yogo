import torch

from copy import deepcopy

from yogo.model import YOGO
from yogo.model_defns import get_model_func


def checkpoint(filepath, net, epoch, global_step):
    "simplified, from yogo.train.Trainer.checkpoint"
    torch.save(
        {
            "epoch": epoch,
            "step": global_step,
            "model_state_dict": deepcopy(net.state_dict()),
            "model_version": net.model_version,
        },
        str(filepath),
    )


def check_model_equality(y, z):
    assert all(yis == zis for yis, zis in zip(y.img_size, z.img_size))
    assert y.anchor_w == z.anchor_w
    assert y.anchor_h == z.anchor_h
    assert y.num_classes == z.num_classes
    assert y.is_rgb == z.is_rgb
    assert y.normalize_images == z.normalize_images
    assert y.clip_value == z.clip_value
    assert y.height_multiplier == z.height_multiplier
    assert y.width_multiplier == z.width_multiplier
    assert y.model_version == z.model_version

    for p1, p2 in zip(y.parameters(), z.parameters()):
        assert p1.data.ne(p2.data).sum() == 0


def test_model_io_basic(tmpdir):
    "shouldn't throw an error"
    y = YOGO(img_size=(772, 1032), anchor_w=0.05, anchor_h=0.05, num_classes=7)
    checkpoint(tmpdir / "test.pth", y, 0, 0)
    z, _ = YOGO.from_pth(tmpdir / "test.pth")
    check_model_equality(y, z)


def test_model_io_different_model_version(tmpdir):
    y = YOGO(
        img_size=(772, 1032),
        anchor_w=0.05,
        anchor_h=0.05,
        num_classes=7,
        model_func=get_model_func("silu_model"),
    )
    checkpoint(tmpdir / "test.pth", y, 0, 0)
    z, _ = YOGO.from_pth(tmpdir / "test.pth")
    check_model_equality(y, z)
    assert y.model_version == "silu_model"
