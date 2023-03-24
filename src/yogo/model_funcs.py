from torch import nn

def model_no_dropout(num_classes) -> nn.Module:
    conv_block_1 = nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False),
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
        nn.BatchNorm2d(16),
    )
    conv_block_4 = nn.Sequential(
        nn.Conv2d(64, 128, 3, padding=1),
        nn.LeakyReLU(),
    )
    conv_block_5 = nn.Sequential(
        nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
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

def model_batchnorm_tweaks(self, num_classes) -> nn.Module:
    conv_block_1 = nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.2),
        nn.BatchNorm2d(16),
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
        nn.LeakyReLU(),
        nn.BatchNorm2d(128),
    )
    conv_block_6 = nn.Sequential(
        nn.Conv2d(128, 128, 3, padding=1, bias=False),
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



def gen_model(num_classes) -> nn.Module:
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
        nn.Conv2d(128, 128, 3, padding=1, bias=False),
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
