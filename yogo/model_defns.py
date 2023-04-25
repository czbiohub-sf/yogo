from typing import Optional, Callable

from torch import nn


def base_model(num_classes) -> nn.Module:
    conv_block_1 = nn.Sequential(
        nn.Conv2d(1, 16, 5, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(16),
        nn.LeakyReLU(),
    )
    conv_block_2 = nn.Sequential(
        nn.Conv2d(16, 32, 3, padding=1),
        nn.LeakyReLU(),
    )
    conv_block_3 = nn.Sequential(
        nn.Conv2d(32, 64, 5, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
    )
    conv_block_4 = nn.Sequential(
        nn.Conv2d(64, 128, 3, padding=1),
        nn.LeakyReLU(),
    )
    conv_block_5 = nn.Sequential(
        nn.Conv2d(128, 128, 5, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
    )
    conv_block_6 = nn.Sequential(
        nn.Conv2d(128, 128, 3, padding=1, bias=True),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
    )
    conv_block_7 = nn.Sequential(
        nn.Conv2d(128, 128, 3, padding=1),
        nn.LeakyReLU(),
    )
    conv_block_8 = nn.Conv2d(128, 5 + num_classes, 1)
    return nn.Sequential(
        conv_block_1,
        conv_block_2,
        conv_block_3,
        conv_block_4,
        conv_block_5,
        conv_block_6,
        conv_block_7,
        conv_block_8,
    )


def model_no_dropout(num_classes: int) -> nn.Module:
    conv_block_1 = nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=2, padding=1),
        nn.LeakyReLU(),
        nn.BatchNorm2d(16),
    )
    conv_block_2 = nn.Sequential(
        nn.Conv2d(16, 32, 3, padding=1),
        nn.LeakyReLU(),
    )
    conv_block_3 = nn.Sequential(
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
        nn.LeakyReLU(),
        nn.BatchNorm2d(64),
    )
    conv_block_4 = nn.Sequential(
        nn.Conv2d(64, 128, 3, padding=1),
        nn.LeakyReLU(),
    )
    conv_block_5 = nn.Sequential(
        nn.Conv2d(128, 128, 3, stride=2, padding=1),
        nn.LeakyReLU(),
        nn.BatchNorm2d(128),
    )
    conv_block_6 = nn.Sequential(
        nn.Conv2d(128, 128, 3, padding=1),
        nn.LeakyReLU(),
        nn.BatchNorm2d(128),
    )
    conv_block_7 = nn.Sequential(
        nn.Conv2d(128, 128, 3, padding=1),
        nn.LeakyReLU(),
    )
    conv_block_8 = nn.Conv2d(128, 5 + num_classes, 1)
    return nn.Sequential(
        conv_block_1,
        conv_block_2,
        conv_block_3,
        conv_block_4,
        conv_block_5,
        conv_block_6,
        conv_block_7,
        conv_block_8,
    )


def model_smaller_SxSy(num_classes: int) -> nn.Module:
    conv_block_1 = nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(16),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.2),
    )
    conv_block_2 = nn.Sequential(
        nn.Conv2d(16, 32, 3, padding=1),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.2),
    )
    conv_block_3 = nn.Sequential(
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.2),
    )
    conv_block_4 = nn.Sequential(
        nn.Conv2d(64, 128, 3, padding=1),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.2),
    )
    conv_block_5 = nn.Sequential(
        nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
    )
    conv_block_6 = nn.Sequential(
        nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
    )
    conv_block_7 = nn.Sequential(
        nn.Conv2d(128, 128, 3, padding=1),
        nn.LeakyReLU(),
    )
    conv_block_8 = nn.Conv2d(128, 5 + num_classes, 1)
    return nn.Sequential(
        conv_block_1,
        conv_block_2,
        conv_block_3,
        conv_block_4,
        conv_block_5,
        conv_block_6,
        conv_block_7,
        conv_block_8,
    )


def model_big_simple(num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2),
            nn.LeakyReLU(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.LeakyReLU(),
        ),
        nn.Sequential(
            nn.Conv2d(32, 32, 3),
            nn.LeakyReLU(),
        ),
        nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2),
            nn.LeakyReLU(),
        ),
        nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.LeakyReLU(),
        ),
        nn.Sequential(
            nn.Conv2d(128, 128, 3),
            nn.LeakyReLU(),
        ),
        nn.Sequential(
            nn.Conv2d(128, 128, 3),
            nn.LeakyReLU(),
        ),
        nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.LeakyReLU(),
        ),
        nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2),
            nn.LeakyReLU(),
        ),
        nn.Sequential(
            nn.Conv2d(256, 256, 3),
            nn.LeakyReLU(),
        ),
        nn.Sequential(
            nn.Conv2d(256, 256, 3),
            nn.LeakyReLU(),
        ),
        nn.Sequential(
            nn.Conv2d(256, 256, 3),
            nn.LeakyReLU(),
        ),
        nn.Conv2d(256, 5 + num_classes, 1),
    )


def model_big_normalized(num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
        ),
        nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.LeakyReLU(),
        ),
        nn.Sequential(
            nn.Conv2d(32, 32, 3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
        ),
        nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2),
            nn.LeakyReLU(),
        ),
        nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
        ),
        nn.Sequential(
            nn.Conv2d(128, 128, 3),
            nn.LeakyReLU(),
        ),
        nn.Sequential(
            nn.Conv2d(128, 128, 3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
        ),
        nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
        ),
        nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2),
            nn.LeakyReLU(),
        ),
        nn.Sequential(
            nn.Conv2d(256, 256, 3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
        ),
        nn.Sequential(
            nn.Conv2d(256, 256, 3),
            nn.LeakyReLU(),
        ),
        nn.Sequential(
            nn.Conv2d(256, 256, 3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
        ),
        nn.Conv2d(256, 5 + num_classes, 1),
    )


def model_big_heavy_normalized(num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
        ),
        nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
        ),
        nn.Sequential(
            nn.Conv2d(32, 32, 3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
        ),
        nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
        ),
        nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
        ),
        nn.Sequential(
            nn.Conv2d(128, 128, 3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
        ),
        nn.Sequential(
            nn.Conv2d(128, 128, 3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
        ),
        nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
        ),
        nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
        ),
        nn.Sequential(
            nn.Conv2d(256, 256, 3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
        ),
        nn.Sequential(
            nn.Conv2d(256, 256, 3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
        ),
        nn.Sequential(
            nn.Conv2d(256, 256, 3),
            nn.LeakyReLU(),
        ),
        nn.Conv2d(256, 5 + num_classes, 1),
    )


def get_model_func(
    model_name: Optional[str],
) -> Optional[Callable[[int,], nn.Module]]:
    models = {
        "base_model": base_model,
        "model_no_dropout": model_no_dropout,
        "model_smaller_SxSy": model_smaller_SxSy,
        "model_big_simple": model_big_simple,
        "model_big_normalized": model_big_normalized,
        "model_big_heavy_normalized": model_big_heavy_normalized,
    }
    try:
        return models[model_name]
    except KeyError:
        print(f"provided model {model_name} doesn't exist; defaulting to None")
        return None
