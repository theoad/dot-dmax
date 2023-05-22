import functools
from typing import Union, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from contextlib import AbstractContextManager


def unfold(
        image: Tensor,  # [b, c, h, w]
        patch_size: Union[Tuple[int, int], Tensor],
        stride: Union[int, Tuple[int, int]]
) -> Tuple[Tensor, int]:
    unfolded = F.unfold(image, patch_size, stride=stride)  # [b, c*ph*pw, n_patches]
    patches = unfolded.transpose(1, 2).unflatten(2, [image.size(1), *patch_size])  # [b, n_patches, c, h, w]
    return patches, unfolded.size(-1)


def fold(
        patches: Tensor,  # [b, n_patches, c, h, w]
        out_size: Union[Tuple[int, int], Tensor],
        patch_size: Union[Tuple[int, int], Tensor],
        stride: Union[int, Tuple[int, int]]
) -> Tensor:
    unfolded = patches.flatten(2).transpose(1, 2)  # [b, c*ph*pw, n_patches]
    out_image = F.fold(unfolded, out_size, patch_size, stride=stride)  # [b, c, h, w]
    return out_image


def image_to_tiles(
        image: Tensor,
        tile_size: Union[Tuple[int, int], Tensor],
        stride: Union[int, Tuple[int, int]]
) -> Tuple[Tensor, Tuple[int, int]]:
    patchify = functools.partial(unfold, patch_size=tile_size, stride=stride)

    size = torch.as_tensor(image.shape[-2:])
    remainder = (size - tile_size) % stride
    mt_size = size - remainder

    t_h, t_w = tile_size
    r_h, r_w = remainder
    mt_h, mt_w = mt_size

    patches, n = patchify(image[..., :mt_h, :mt_w])
    bottom_patches, n_bottom = patchify(image[..., -t_h:, :mt_w])
    right_patches, n_right = patchify(image[..., :mt_h, -t_w:])
    corner = image[:, None, :, -t_h:, -t_w:]

    if r_h:
        patches = torch.cat([patches, bottom_patches], dim=1)
    if r_w:
        patches = torch.cat([patches, right_patches], dim=1)
    if r_h and r_w:
        patches = torch.cat([patches, corner], dim=1)

    return patches, (n_bottom if r_h else 0, n_right if r_w else 0)


def tiles_to_image(
        tiles: Tensor,
        out_size: Union[Tuple[int, int], Tensor],
        tile_size: Union[Tuple[int, int], Tensor],
        stride: Union[int, Tuple[int, int]],
        borders: Tuple[int, int],
):
    unpatchify = functools.partial(fold, patch_size=tile_size, stride=stride)
    image = torch.zeros(tiles.size(0), tiles.size(2), *out_size, device=tiles.device, dtype=tiles.dtype)

    n_bottom, n_right = borders

    size = torch.as_tensor(image.shape[-2:])
    remainder = (size - tile_size) % stride
    mt_size = size - remainder

    t_h, t_w = tile_size
    mt_h, mt_w = mt_size

    if n_bottom and n_right:
        tiles, corner = tiles[:, :-1], tiles[:, -1]
        image[..., -t_h:, -t_w:] += corner
    if n_right:
        tiles, right_border = tiles[:, :-n_right], tiles[:, -n_right:]
        image[..., :mt_h, -t_w:] += unpatchify(right_border, image[..., :mt_h, -t_w:].shape[-2:])
    if n_bottom:
        tiles, bottom_border = tiles[:, :-n_bottom], tiles[:, -n_bottom:]
        image[..., -t_h:, :mt_w] += unpatchify(bottom_border, image[..., -t_h:, :mt_w].shape[-2:])

    image[..., :mt_h, :mt_w] += unpatchify(tiles, mt_size)
    return image


class Tile(AbstractContextManager):
    """
    with Tile(image, tile_size=64, overlap=3/4) as T:
        tiles = T.tiles
        processed = do_stuff(tiles)   # work on patches
        out = T.fold_back(processed)  # average back the overlapping patches
    """
    def __init__(self, image: torch.Tensor, tile_size: Union[int, Tuple[int, int]] = 512, overlap: float = 0.25):
        self.image = image
        self.tiles = None
        self.borders = None

        self.tile_size = torch.as_tensor(tile_size if isinstance(tile_size, tuple) else (tile_size, tile_size))
        self.overlap = overlap

        self.size = torch.as_tensor(image.shape[-2:])
        self.tile = torch.min(self.tile_size, self.size)
        self.stride = self.tile - (self.tile * overlap).int()

    def __enter__(self):
        if (self.size <= self.tile_size).all():
            self.tiles = self.image
        # image to batch of tiles
        self.tiles, self.borders = image_to_tiles(self.image, self.tile, self.stride)  # [b, c, h, w] --> [b, n_patches, c, ph, pw]
        self.tiles = self.tiles.flatten(0, 1)  # [b, n_patches, c, ph, pw] --> [b * n_patches, c, ph, pw]
        return self

    def fold_back(self, tiles: torch.Tensor) -> torch.Tensor:
        out_tile = torch.as_tensor(tiles.shape[-2:])
        out_size = (self.size * out_tile / self.tile).int()  # infer scaling factor from the output
        out_stride = out_tile - (out_tile * self.overlap).int()

        # tiles back to image
        out_patches = tiles.unflatten(0, [self.image.size(0), -1])  # [b * n_patches, oc, oph, opw] --> [b, n_patches, oc, oph, opw]
        out_image = tiles_to_image(out_patches, out_size, out_tile, out_stride, self.borders)  # [b, oc, oh, ow]
        divisor = tiles_to_image(torch.ones_like(out_patches), out_size, out_tile, out_stride, self.borders)
        return out_image / divisor

    def __exit__(self, *args):
        return False


def tiled(tile_size: Union[int, Tuple[int, int]] = 512, overlap: float = 0.25):
    """
    Method decorator to allow tiled computation of large images (with resolution >  tile_size)
    convenient to ensure constant memory usage even when evaluating large images.

    Usage:

        @tiled(max_res=512, overlap=0.25)
        def forward(self, image: torch.Tensor, ...):
            ...  # regular implementation

    Note: Although this implementation is differentiable, it was not designed to preserve gradients.

    :param tile_size: The maximum resolution of the tile (it is trimmed with the image shortest edge).
    :param overlap: Overlap ratio between tiles to avoid blockiness artifacts.
    """
    def tiled_forward(forward):
        @functools.wraps(forward)
        def wrapper(self, image, **kwargs):
            with Tile(image, tile_size, overlap) as T:
                return T.fold_back(forward(self, T.tiles, **kwargs))
        return wrapper
    return tiled_forward


if __name__ == '__main__':

    def image1234(image):
        sf = 1
        b, c, h, w = image.size()
        ones = torch.ones(b, h * sf, w * sf)
        return torch.stack([ones * 1, ones * 2, ones * 3], dim=1)  # emulates a 4-channel upscaled output


    class Dummy(torch.nn.Module):
        @tiled(tile_size=256, overlap=0.25)
        def forward(self, image):
            return image1234(image)

    dummy = Dummy()
    input = torch.randn(1, 3, 448, 736)  # input not multiple of tile size
    assert (dummy(input) == image1234(input)).all()
