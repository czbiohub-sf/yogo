import torch
from torch import nn

from pathlib import Path
from typing import Tuple, Optional, Callable, Union, Any, Dict

from yogo.model_defns import get_model_func


class YOGO(nn.Module):
    """
    Restricting assumptions:
        - all objects being detected are roughly the same size (K-Means Clustering anchor
        boxes across the dataset verifies this), meaning that it does not make sense to
        have more than 1 anchor box
        - grayscale

    TODO: We do messy stuff here w/r/t inference vs. training mode. While training, we
    want to set self.training=False occasionally (batchnorm and dropout behaviour is
    different during inference), but we still want to use the yogo_loss function to
    measure validation and test, so we do not want to convert to sigmoids or whatever
    else.

    A better way to do this would be to have an "inference" method, that you could plug
    onto the end of forward if we are running inference.
    """

    def __init__(
        self,
        img_size: Tuple[int, int],
        anchor_w: float,
        anchor_h: float,
        num_classes: int,
        inference: bool = False,
        tuning: bool = False,
        model_func: Optional[
            Callable[
                [
                    int,
                ],
                nn.Module,
            ]
        ] = None,
        device: Union[torch.device, str] = "cpu",
    ):
        super().__init__()
        self.device = device

        self.model = (
            self.gen_model(num_classes=num_classes)
            if model_func is None
            else model_func(num_classes)
        )

        self.register_buffer("img_size", torch.tensor(img_size))
        self.register_buffer("anchor_w", torch.tensor(anchor_w))
        self.register_buffer("anchor_h", torch.tensor(anchor_h))
        self.register_buffer("num_classes", torch.tensor(num_classes))

        self.inference = inference

        Sx, Sy = self.get_grid_size()

        _Cxs = torch.linspace(0, 1 - 1 / Sx, Sx).expand(Sy, -1).to(self.device)
        _Cys = (
            torch.linspace(0, 1 - 1 / Sy, Sy)
            .expand(1, -1)
            .transpose(0, 1)
            .expand(Sy, Sx)
            .to(self.device)
        )

        # this feels wrong, but it works - there is some issue
        # with just giving _Cxs / _Cys directly when initting via
        # from_pth
        self.register_buffer("_Cxs", _Cxs.clone())
        self.register_buffer("_Cys", _Cys.clone())

        # multiplier for height - req'd when resizing model post-training
        self.register_buffer("height_multiplier", torch.tensor(1.0))

        # initialize the weights, PyTorch chooses bad defaults
        self.model.apply(self.init_network_weights)

        # fine tuning. If you `.eval()` the model anyways, this
        # is not necessary
        if tuning:
            self.model.apply(self.set_bn_eval)

    @staticmethod
    def init_network_weights(module: nn.Module):
        if isinstance(module, nn.Conv2d):
            # init weights to default leaky relu neg slope, biases to 0
            torch.nn.init.kaiming_normal_(
                module.weight, a=0.01, mode="fan_out", nonlinearity="leaky_relu"
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    @staticmethod
    def set_bn_eval(module: nn.Module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    @classmethod
    def from_pth(
        cls, pth_path: Path, inference: bool = False
    ) -> Tuple["YOGO", Dict[str, Any]]:
        loaded_pth = torch.load(pth_path, map_location="cpu")

        model_version = loaded_pth.get("model_version", None)
        global_step = loaded_pth.get("step", 0)
        normalize_images = loaded_pth.get("normalize_images", False)

        params = loaded_pth["model_state_dict"]
        img_size = params["img_size"]
        anchor_w = params["anchor_w"]
        anchor_h = params["anchor_h"]
        num_classes = params["num_classes"]

        if "height_multiplier" not in params:
            params["height_multiplier"] = torch.tensor(1.0)

        model = cls(
            (img_size[0], img_size[1]),
            anchor_w.item(),
            anchor_h.item(),
            num_classes=num_classes.item(),
            inference=inference,
            model_func=get_model_func(model_version),
        )

        model.load_state_dict(params)

        return model, {
            "step": global_step,
            "normalize_images": normalize_images,
        }

    def to(self, device, *args, **kwargs):
        self.device = device
        super().to(device, *args, **kwargs)
        return self

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def grad_norm(self) -> float:
        # https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961/5
        total_norm = 0
        parameters = [
            p for p in self.parameters() if p.grad is not None and p.requires_grad
        ]
        for p in parameters:
            if p.grad is None:
                continue
            gradient_norm = p.grad.detach().data.norm(2)
            total_norm += gradient_norm.item() ** 2
        total_norm = total_norm**0.5
        return total_norm

    def param_norm(self) -> float:
        # https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961/5
        total_norm = 0
        parameters = [
            p for p in self.parameters() if p.grad is not None and p.requires_grad
        ]
        for p in parameters:
            parameter_norm = p.detach().data.norm(2)
            total_norm += parameter_norm.item() ** 2
        total_norm = total_norm**0.5
        return total_norm

    def get_img_size(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.img_size, torch.Tensor):
            h, w = self.img_size
            return h, w
        raise ValueError(f"self.img_size is not a tensor: {type(self.img_size)}")

    def get_grid_size(
        self, img_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[int, int]:
        """return Sx, Sy

        We could support arbitrary layers, but that would take a long time, and
        would be overcmoplicated for what we are doing - we can add modules
        here as we add different types of layers
        """
        if img_size is not None:
            # appease type checker
            h, w = torch.tensor(img_size)
        else:
            h, w = self.get_img_size()

        def as_tuple(inp: Union[Any, Tuple[Any, Any]]) -> Tuple[Any, ...]:
            return inp if isinstance(inp, tuple) else (inp, inp)

        for mod in self.modules():
            if isinstance(
                mod,
                nn.Conv2d,
            ):
                if isinstance(mod.padding, tuple):
                    p0, p1 = mod.padding
                elif mod.padding is None or mod.padding == "none":
                    p0, p1 = 0, 0
                d0, d1 = as_tuple(mod.dilation)
                k0, k1 = as_tuple(mod.kernel_size)
                s0, s1 = as_tuple(mod.stride)
                h = torch.floor((h + 2 * p0 - d0 * (k0 - 1) - 1) / s0 + 1)
                w = torch.floor((w + 2 * p1 - d1 * (k1 - 1) - 1) / s1 + 1)

        Sy = h.item()
        Sx = w.item()
        # type checker is unhappy if I `h.int().item()` instead of
        # int(h.item())
        return int(Sx), int(Sy)

    def resize_model(self, img_height: int) -> None:
        """
        for YOGO's specific application of counting cells as they flow
        past a FOV, it is useful to take a crop of the images in order
        to reduce double-counting cells. This function resizes the
        model to a certain image height - 193 px is about a quarter
        of the full 772 pixel height, and is standard for our uses.
        """
        org_img_height, org_img_width = (int(d) for d in self.get_img_size())
        crop_size = (img_height, org_img_width)
        Sx, Sy = self.get_grid_size(crop_size)
        _Cxs = torch.linspace(0, 1 - 1 / Sx, Sx, device=self.device).expand(Sy, -1)
        _Cys = (
            torch.linspace(0, 1 - 1 / Sy, Sy, device=self.device)
            .expand(1, -1)
            .transpose(0, 1)
            .expand(Sy, Sx)
        )
        self.register_buffer(
            "height_multiplier", torch.tensor(org_img_height / img_height)
        )
        self.register_buffer("img_size", torch.tensor(crop_size))
        self.register_buffer("_Cxs", _Cxs.clone())
        self.register_buffer("_Cys", _Cys.clone())

    def gen_model(self, num_classes) -> nn.Module:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # we get either raw uint8 tensors or float tensors
        x = self.model(x.float())

        _, _, Sy, Sx = x.shape

        if self.inference:
            classification = torch.softmax(x[:, 5:, :, :], dim=1)
        else:
            classification = x[:, 5:, :, :]

        # implementation of "Direct Location Prediction" from YOLO9000 paper
        #  center of bounding box in x
        #  center of bounding box in y
        #  width of bounding box
        #  height of bounding box
        #  'objectness' score
        return torch.cat(
            (
                (1 / Sx) * torch.sigmoid(x[:, 0:1, :, :]) + self._Cxs,
                (1 / Sy) * torch.sigmoid(x[:, 1:2, :, :]) + self._Cys,
                self.anchor_w * torch.exp(x[:, 2:3, :, :]),
                self.anchor_h * torch.exp(x[:, 3:4, :, :]) * self.height_multiplier,
                torch.sigmoid(x[:, 4:5, :, :]),
                *torch.split(classification, 1, dim=1),
            ),
            dim=1,
        )
