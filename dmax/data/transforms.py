from typing import Literal
import torch
import torch.nn as nn
from torchvision.io.image import encode_jpeg, decode_jpeg
import torchvision.transforms.functional as F

__all__ = ["ToMultiple", "Downsample", "AddNoise", "Jpeg"]


class ToMultiple(nn.Module):
    def __init__(self, multiple_of: int = 8, mode: Literal["pad", "crop"] = "pad"):
        super().__init__()
        self.multiple_of = multiple_of
        self.mode = mode

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        height, width = input.shape[-2:]
        if self.mode == "pad":
            H = ((height + self.multiple_of) // self.multiple_of) * self.multiple_of
            W = ((width + self.multiple_of) // self.multiple_of) * self.multiple_of
            padh = int(H - height) if height % self.multiple_of != 0 else 0
            padw = int(W - width) if width % self.multiple_of != 0 else 0
            padded = F.pad(input, [0, 0, padw, padh], padding_mode='reflect')  # noqa
            return padded
        elif self.mode == "crop":
            H = (height // self.multiple_of) * self.multiple_of
            W = (width // self.multiple_of) * self.multiple_of
            cropped = F.center_crop(input, [H, W])
            return cropped

    def extra_repr(self) -> str:
        return f"multiple_of={str(self.multiple_of)}"


class Downsample(nn.Module):
    def __init__(self, scale: int = 4, upsample_back: bool = False):
        super().__init__()
        self.scale = scale
        self.upsample_back = upsample_back

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        h, w = input.shape[-2:]
        down = F.resize(input, [h // self.scale, w // self.scale], antialias=True)
        if self.upsample_back:
            down = F.resize(down, [h, w])
        return down

    def extra_repr(self):
        return f"scale={self.scale}, upsample_back={self.upsample_back}"


class AddNoise(nn.Module):
    def __init__(self, sigma: float = 50):
        super().__init__()
        self.sigma = sigma / 255.

    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
        return input + torch.randn_like(input) * self.sigma

    def extra_repr(self):
        return f"sigma={self.sigma}"


class Jpeg(nn.Module):
    def __init__(self, quality: int = 50):
        super().__init__()
        self.quality = quality

    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
        jpeg_file = encode_jpeg((input * 255).byte(), self.quality)
        jpeg_img = decode_jpeg(jpeg_file)
        return jpeg_img.float() / 255.

    def extra_repr(self) -> str:
        return f"quality={self.quality}"


class MaxResResizeCrop(nn.Module):
    def __init__(self, max_resolution: int = 1024):
        super().__init__()
        self.max_resolution = max_resolution

    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
        h, w = input.shape[-2:]
        if min(h, w) > self.max_resolution:
            input = F.resize(input, self.max_resolution)  # noqa
        h, w = input.shape[-2:]
        if max(h, w) > self.max_resolution:
            input = F.center_crop(input, self.max_resolution)  # noqa
        return input

    def extra_repr(self) -> str:
        return f"max_resolution={self.max_resolution}"


class CenterCropLongEdge(nn.Module):
    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
        return F.center_crop(input, min(input.shape[-2:]))  # noqa
