from torch import nn

from typing import Callable, Optional, Dict


ModelDefn = Callable[[int], nn.Module]

MODELS: Dict[str, ModelDefn] = {}


def get_model_func(model_name: Optional[str]) -> ModelDefn:
    if model_name is None:
        return base_model

    try:
        return MODELS[model_name]
    except KeyError:
        return base_model


def register_model(model_defn: ModelDefn) -> ModelDefn:
    """
    put model in MODELS. When adding a new model,
    so make sure to `@register_model`!
    """
    MODELS[model_defn.__name__] = model_defn
    return model_defn


@register_model
def base_model(num_classes: int) -> nn.Module:
    conv_block_1 = nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(16),
        nn.LeakyReLU(),
    )
    conv_block_2 = nn.Sequential(
        nn.Conv2d(16, 32, 3, padding=1),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.05),
    )
    conv_block_3 = nn.Sequential(
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.1),
    )
    conv_block_4 = nn.Sequential(
        nn.Conv2d(64, 128, 3, padding=1),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.15),
    )
    conv_block_5 = nn.Sequential(
        nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
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


@register_model
def double_filters(num_classes: int) -> nn.Module:
    conv_block_1 = nn.Sequential(
        nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
    )
    conv_block_2 = nn.Sequential(
        nn.Conv2d(32, 64, 3, padding=1),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.05),
    )
    conv_block_3 = nn.Sequential(
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.1),
    )
    conv_block_4 = nn.Sequential(
        nn.Conv2d(128, 256, 3, padding=1),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.15),
    )
    conv_block_5 = nn.Sequential(
        nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(),
    )
    conv_block_6 = nn.Sequential(
        nn.Conv2d(256, 256, 3, padding=1, bias=True),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(),
    )
    conv_block_7 = nn.Sequential(
        nn.Conv2d(256, 256, 3, padding=1),
        nn.LeakyReLU(),
    )
    conv_block_8 = nn.Conv2d(256, 5 + num_classes, 1)
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


@register_model
def triple_filters(num_classes: int) -> nn.Module:
    conv_block_1 = nn.Sequential(
        nn.Conv2d(1, 48, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(48),
        nn.LeakyReLU(),
    )
    conv_block_2 = nn.Sequential(
        nn.Conv2d(48, 96, 3, padding=1),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.05),
    )
    conv_block_3 = nn.Sequential(
        nn.Conv2d(96, 192, 3, stride=2, padding=1),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.1),
    )
    conv_block_4 = nn.Sequential(
        nn.Conv2d(192, 384, 3, padding=1),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.15),
    )
    conv_block_5 = nn.Sequential(
        nn.Conv2d(384, 384, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
    )
    conv_block_6 = nn.Sequential(
        nn.Conv2d(384, 384, 3, padding=1, bias=True),
        nn.BatchNorm2d(384),
        nn.LeakyReLU(),
    )
    conv_block_7 = nn.Sequential(
        nn.Conv2d(384, 384, 3, padding=1),
        nn.LeakyReLU(),
    )
    conv_block_8 = nn.Conv2d(384, 5 + num_classes, 1)
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


@register_model
def half_filters(num_classes: int) -> nn.Module:
    conv_block_1 = nn.Sequential(
        nn.Conv2d(1, 8, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(8),
        nn.LeakyReLU(),
    )
    conv_block_2 = nn.Sequential(
        nn.Conv2d(8, 16, 3, padding=1),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.05),
    )
    conv_block_3 = nn.Sequential(
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.1),
    )
    conv_block_4 = nn.Sequential(
        nn.Conv2d(32, 64, 3, padding=1),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.15),
    )
    conv_block_5 = nn.Sequential(
        nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
    )
    conv_block_6 = nn.Sequential(
        nn.Conv2d(64, 64, 3, padding=1, bias=True),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
    )
    conv_block_7 = nn.Sequential(
        nn.Conv2d(64, 64, 3, padding=1),
        nn.LeakyReLU(),
    )
    conv_block_8 = nn.Conv2d(64, 5 + num_classes, 1)
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


@register_model
def quarter_filters(num_classes: int) -> nn.Module:
    conv_block_1 = nn.Sequential(
        nn.Conv2d(1, 4, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(4),
        nn.LeakyReLU(),
    )
    conv_block_2 = nn.Sequential(
        nn.Conv2d(4, 8, 3, padding=1),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.05),
    )
    conv_block_3 = nn.Sequential(
        nn.Conv2d(8, 16, 3, stride=2, padding=1),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.1),
    )
    conv_block_4 = nn.Sequential(
        nn.Conv2d(16, 32, 3, padding=1),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.15),
    )
    conv_block_5 = nn.Sequential(
        nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
    )
    conv_block_6 = nn.Sequential(
        nn.Conv2d(32, 32, 3, padding=1, bias=True),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
    )
    conv_block_7 = nn.Sequential(
        nn.Conv2d(32, 32, 3, padding=1),
        nn.LeakyReLU(),
    )
    conv_block_8 = nn.Conv2d(32, 5 + num_classes, 1)
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


@register_model
def convnext_small(num_classes: int) -> nn.Module:
    try:
        import timm
    except ImportError:
        raise ImportError("Please install timm to use convnext_small.")

    # timm is amazing
    # TODO is it better starting from pretrained? almost certianly yes
    model = timm.create_model(
        "convnext_small", pretrained=False, num_classes=0, in_chans=1
    )

    # we need to replace the last block of this model so the output
    # tensor will match the YOGO format (5 + num_classes, grid_x, grid_y)
    # the last two children are an Identity block and the classification block
    model_chopped = nn.Sequential(*list(model.children())[:-2])

    format_block = nn.Sequential(
        nn.Conv2d(768, 5 + num_classes, 1),
        nn.ConvTranspose2d(5 + num_classes, 5 + num_classes, kernel_size=4, stride=4),
    )

    model_chopped.add_module("format time!", format_block)

    return model_chopped
