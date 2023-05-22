from typing import Optional, Tuple, Callable, List, Union, Dict

import torch
from torch import Tensor, FloatTensor
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import RandomResizedCrop, Resize
from torchvision.transforms.functional import to_tensor, resize, center_crop
from dmax.data.transforms import ToMultiple, MaxResResizeCrop, CenterCropLongEdge

__all__ = ["concat_images_with_pad", "RestorationDataset", "ConcatRestorationDataset"]


def concat_images_with_pad(images: Tuple[Tensor, ...]) -> Tensor:
    b = len(images)
    max_c = max(map(lambda t: t.size(0), images))
    max_h = max(map(lambda t: t.size(1), images))
    max_w = max(map(lambda t: t.size(2), images))

    if all(max_c == img.size(0) for img in images) and \
       all(max_h == img.size(1) for img in images) and \
       all(max_w == img.size(2) for img in images):
        return torch.stack(images)

    images_padded = torch.zeros(b, max_c, max_h, max_w, dtype=images[0].dtype)
    for i, image in enumerate(images):
        c, h, w = image.size()
        images_padded[i, :c, :h, :w] = image
    return images_padded


class RestorationDataset(Dataset):
    Item = Union[Tuple[Tensor, ...], Dict[str, FloatTensor]]

    def __init__(
            self,
            clean_dataset: ConcatDataset,
            degraded_dataset: Optional[ImageFolder] = None,
            *,
            degradation_func: Optional[Callable[[Tensor], Tensor]] = None,
            resolution: Optional[int] = None,
            max_resolution: Optional[int] = None,
            random_crop: bool = False,
            return_dict: bool = False
    ):
        if degraded_dataset is not None:
            assert len(degraded_dataset) == len(clean_dataset),\
                "clean dataset and degraded datasets should have the same length"
        assert degraded_dataset or degradation_func

        self.clean_dataset = clean_dataset
        self.degraded_dataset = degraded_dataset
        self.degradation = degradation_func
        self._resolution = resolution
        self.return_dict = return_dict
        self.random_crop = random_crop
        self.max_resize = MaxResResizeCrop(max_resolution) if max_resolution else None

    def __getitem__(self, index: int) -> Item:
        image = self.clean_dataset[index]

        if isinstance(image, (tuple, list)): image = image[0]
        if not isinstance(image, Tensor): image = to_tensor(image)

        if self.degraded_dataset is None:
            degraded = image.clone()
        else:
            degraded = self.degraded_dataset[index]

            if isinstance(degraded, (tuple, list)):
                degraded = degraded[0]

            if not isinstance(degraded, Tensor):
                degraded = to_tensor(degraded)

        if self.max_resize is not None:
            image, degraded = self.apply_transform(self.max_resize, image, degraded)

        if image.shape[-2:] != degraded.shape[-2:] and self.resolution:
            image, degraded = center_crop(image, min(image.shape[-2:])), center_crop(degraded, min(degraded.shape[-2:]))
            scale = int(max(image.shape[-1] / degraded.shape[-1], image.shape[-2] / degraded.shape[-2]))
            degraded = Resize((self.resolution // scale,) * 2)(degraded)
            image = Resize((self.resolution,) * 2)(image)
        else:
            image, degraded = self.apply_transform(self.resize, image, degraded)

        if self.degradation is not None:
            degraded = self.degradation(degraded)

        if self.return_dict:
            return {'degraded': degraded, 'clean': image}

        return degraded, image

    def __len__(self):
        return len(self.clean_dataset)

    @staticmethod
    def apply_transform(transform, image, degraded):
        if isinstance(transform, RandomResizedCrop):
            # concat on 1st dimension before applying transforms
            # This way we ensure the degraded and clean are cropped in the same place.
            image, degraded = torch.chunk(transform(torch.cat([image, degraded], dim=0)), 2, dim=0)
        else:
            image, degraded = transform(image), transform(degraded)
        return image, degraded

    @property
    def resolution(self) -> int:
        return self._resolution

    @resolution.setter
    def resolution(self, value: Optional[int]):
        self._resolution = value

    @property
    def resize(self) -> Callable[[Tensor], Tensor]:
        if self.resolution is None:
            return ToMultiple(8)
        if self.random_crop:
            return RandomResizedCrop(self.resolution, antialias=False)
        else:
            return torch.nn.Sequential(CenterCropLongEdge(), Resize((self.resolution,) * 2, antialias=False))

    @staticmethod
    def collate_fn(batch: List[Item]) -> Item:
        if all(isinstance(elem, tuple) for elem in batch):
            return tuple(concat_images_with_pad(images) for images in zip(*batch))            # list of tuple to tuple of list(as tensor)

        return {k: concat_images_with_pad(tuple(dic[k] for dic in batch)) for k in batch[0]}  # list of dict to dict of list(as tensor)


class ConcatRestorationDataset(ConcatDataset):
    datasets: List[RestorationDataset]
    collate_fn = RestorationDataset.collate_fn

    @property
    def resolution(self):
        return self.datasets[0].resolution

    @resolution.setter
    def resolution(self, value: Optional[int]):
        for dataset in self.datasets:
            dataset.resolution = value
