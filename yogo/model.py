import torch
from torch import nn

from pathlib import Path
from typing import Tuple, Optional, Union, Any, Dict

from yogo.model_defns import ModelDefn, base_model, get_model_func


PathLike = Union[Path, str]


class YOGO(nn.Module):
    def __init__(
        self,
        img_size: Tuple[int, int],
        anchor_w: float,
        anchor_h: float,
        num_classes: int,
        is_rgb: bool = False,
        normalize_images: bool = False,
        inference: bool = False,
        tuning: bool = False,
        model_func: ModelDefn = base_model,
        clip_value: float = 1.0,
        device: Union[torch.device, str] = "cpu",
    ):
        super().__init__()

        self.device = device

        self.model = model_func(num_classes, is_rgb).to(device)
        self.model_version = model_func.__name__

        self.register_buffer("img_size", torch.tensor(img_size))
        self.register_buffer("anchor_w", torch.tensor(anchor_w))
        self.register_buffer("anchor_h", torch.tensor(anchor_h))
        self.register_buffer("num_classes", torch.tensor(num_classes))
        self.register_buffer("clip_value", torch.tensor(clip_value))
        self.register_buffer("is_rgb", torch.tensor(is_rgb))
        self.register_buffer("normalize_images", torch.tensor(normalize_images))

        self.inference = inference

        Sx, Sy = self.get_grid_size()
        self.Sx, self.Sy = Sx, Sy

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
        self.register_buffer("width_multiplier", torch.tensor(1.0))

        # fine tuning. If you `.eval()` the model anyways, this
        # is not necessary
        if tuning:
            self.model.apply(self.set_bn_eval)
        else:
            # initialize the weights, PyTorch chooses bad defaults
            self.model.apply(self.init_network_weights)

        # gradient clipping
        for p in self.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

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
        cls, pth_path: PathLike, inference: bool = False
    ) -> Tuple["YOGO", Dict[str, Any]]:
        pth_path = Path(pth_path)
        loaded_pth = torch.load(pth_path, map_location="cpu")

        global_step = loaded_pth.get("step", 0)
        model_version = loaded_pth.get("model_version", None)
        class_names = loaded_pth.get("class_names", None)

        params = loaded_pth["model_state_dict"]
        img_size = params["img_size"]
        anchor_w = params["anchor_w"]
        anchor_h = params["anchor_h"]
        num_classes = params["num_classes"]

        # be permissive of older pth files
        if "is_rgb" not in params:
            params["is_rgb"] = torch.tensor(False)

        if "clip_value" not in params:
            params["clip_value"] = torch.tensor(1.0)

        if "height_multiplier" not in params:
            params["height_multiplier"] = torch.tensor(1.0)

        if "width_multiplier" not in params:
            params["width_multiplier"] = torch.tensor(1.0)

        if "normalize_images" not in params:
            normalize_images = loaded_pth.get("normalize_images", False)
            params["normalize_images"] = torch.tensor(normalize_images)

        model = cls(
            (img_size[0], img_size[1]),
            anchor_w.item(),
            anchor_h.item(),
            num_classes=num_classes.item(),
            inference=inference,
            tuning=True,
            model_func=get_model_func(model_version),
        )

        model.load_state_dict(params)

        if inference:
            model.eval()

        return model, {
            "step": global_step,
            "class_names": class_names,
            "normalize_images": params["normalize_images"],
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
            if isinstance(inp, tuple):
                return inp
            elif inp is None or inp == "none":
                # TODO is this the right default?
                return (0, 0)
            return (inp, inp)

        for mod in self.modules():
            if isinstance(
                mod,
                nn.Conv2d,
            ):
                p0, p1 = as_tuple(mod.padding)
                d0, d1 = as_tuple(mod.dilation)
                k0, k1 = as_tuple(mod.kernel_size)
                s0, s1 = as_tuple(mod.stride)
                h = torch.floor((h + 2 * p0 - d0 * (k0 - 1) - 1) / s0 + 1)
                w = torch.floor((w + 2 * p1 - d1 * (k1 - 1) - 1) / s1 + 1)
            elif isinstance(mod, nn.ConvTranspose2d):
                p0, p1 = as_tuple(mod.padding)
                d0, d1 = as_tuple(mod.dilation)
                k0, k1 = as_tuple(mod.kernel_size)
                s0, s1 = as_tuple(mod.stride)
                p_o0, p_o1 = as_tuple(mod.output_padding)
                h = torch.floor((h - 1) * s0 - 2 * p0 + d0 * (k0 - 1) + p_o0 + 1)
                w = torch.floor((w - 1) * s1 - 2 * p1 + d1 * (k1 - 1) + p_o1 + 1)

        Sy = h.item()
        Sx = w.item()
        return int(Sx), int(Sy)

    def resize_model(
        self, img_height: Optional[int] = None, img_width: Optional[int] = None
    ) -> None:
        """
        for YOGO's specific application of counting cells as they flow
        past a FOV, it is useful to take a crop of the images in order
        to reduce double-counting cells. This function resizes the
        model to a certain image height - 193 px is about a quarter
        of the full 772 pixel height, and is standard for our uses.
        """
        org_img_height, org_img_width = (int(d) for d in self.get_img_size())
        crop_size = (img_height or org_img_height, img_width or org_img_width)
        Sx, Sy = self.get_grid_size(crop_size)
        self.Sx, self.Sy = Sx, Sy
        _Cxs = torch.linspace(0, 1 - 1 / Sx, Sx, device=self.device).expand(Sy, -1)
        _Cys = (
            torch.linspace(0, 1 - 1 / Sy, Sy, device=self.device)
            .expand(1, -1)
            .transpose(0, 1)
            .expand(Sy, Sx)
        )
        self.register_buffer(
            "height_multiplier", torch.tensor(org_img_height / crop_size[0])
        )
        self.register_buffer(
            "width_multiplier", torch.tensor(org_img_width / crop_size[1])
        )
        self.register_buffer("img_size", torch.tensor(crop_size))
        self.register_buffer("_Cxs", _Cxs.clone())
        self.register_buffer("_Cys", _Cys.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # we get either raw uint8 tensors or float tensors
        if x.ndim == 3:
            x.unsqueeze_(0)

        if not x.is_floating_point():
            x = x.float()

        x = self.model(x)

        _, _, Sy, Sx = x.shape

        if self.inference:
            classification = torch.softmax(x[:, 5:, :, :], dim=1)
        else:
            classification = x[:, 5:, :, :]

        # torch.exp(89) == tensor(inf)!
        # so clamp the max value to 80 (for good measure)
        # torch.exp(80) == 5.5406e+34, so that's plenty.
        clamped_whs = torch.clamp(x[:, 2:4, :, :], max=80)

        # implementation of "Direct Location Prediction" from YOLO9000 paper
        #  center of bounding box in x
        #  center of bounding box in y
        #  width of bounding box
        #  height of bounding box
        #  'objectness' score
        return torch.cat(
            (
                ((1 / Sx) * torch.sigmoid(x[:, 0, :, :]) + self._Cxs)[:, None, :, :],
                ((1 / Sy) * torch.sigmoid(x[:, 1, :, :]) + self._Cys)[:, None, :, :],
                (
                    self.anchor_w
                    * torch.exp(clamped_whs[:, 0:1, :, :])
                    * self.width_multiplier
                ),
                (
                    self.anchor_h
                    * torch.exp(clamped_whs[:, 1:2, :, :])
                    * self.height_multiplier
                ),
                (torch.sigmoid(x[:, 4, :, :]))[:, None, :, :],
                *torch.split(classification, 1, dim=1),
            ),
            dim=1,
        )
