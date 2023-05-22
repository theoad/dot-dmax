import re
import os
from typing import Union, Literal, Optional, Tuple, Callable
from PIL import Image
from datetime import datetime

from torch.utils.data import Dataset, ConcatDataset, Subset, random_split, DataLoader
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.datasets import ImageNet
import torchvision.transforms as T

from dmax.data.gdrive import *
from dmax.data.transforms import *
from dmax.data.restoration_dataset import *
from dmax.utils.ddp import rank_zero_first

PAIRED_DATASETS = ["RealBlurR", "RealBlurJ", "HIDE", "GoPro", "rain13k"]


class Folder(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None) -> None:
        self.root = os.path.expanduser(root)
        self.img_paths = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            if dirnames: continue
            self.img_paths.extend([
                os.path.join(dirpath, filename)
                for filename in sorted(filenames)
                if os.path.splitext(filename)[1].lower() in IMG_EXTENSIONS
            ])
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index: int):
        img = Image.open(self.img_paths[index])
        if self.transform is not None:
            img = self.transform(img)
        return img


class ImageNetNoLabel(ImageNet):
    def __getitem__(self, index: int):
        return super().__getitem__(index)[0]


def concat_folders(root, transform, dirpath_filter = lambda p: True):
    folders = []
    for dirpath, dirnames, _ in os.walk(root):
        if not dirnames and dirpath_filter(dirpath):
            data = Folder(dirpath, transform)
            folders.append(data)
    return ConcatDataset(folders)


def dataset(
        dataset_name: str,
        split: Optional[Literal["train", "validation"]] = "train",
        paired: bool = False,
        dataset_url: Optional[str] = None,
        subfolder: Optional[Union[str, Tuple[str, str]]] = None,
        train_val_ratio: float = 0.9,
        transform: Optional[Callable] = None
):
    if '-' in dataset_name:
        dataset_name, split = dataset_name.split('-')
    if "imagenet" in dataset_name.lower():
        return ImageNetNoLabel("~/data/ImageNet", split="val" if split == "val" else split, transform=transform),
    elif "set" in dataset_name.lower():
        subfolder = subfolder or "original"

    paired = paired or (subfolder is not None and isinstance(subfolder, tuple))
    root = os.path.join(DATASET_CACHE, dataset_name)
    if not os.path.exists(root): download(dataset_name, absolute_url=dataset_url)

    need_split = split is not None
    if split is not None and os.path.exists(os.path.join(root, split)):
        root = os.path.join(root, split)
        need_split = False

    def subset(ds):
        if not need_split: return ds
        train_size = int(len(ds) * train_val_ratio)
        split_idx = range(train_size) if split == "train" else range(train_size, len(ds))
        return Subset(ds, split_idx)

    if not paired:
        filter = lambda dirpath: dirpath.endswith(subfolder) if subfolder else lambda p: True
        return subset(concat_folders(root, transform, dirpath_filter=filter)),

    subfolder = subfolder or ("target", "input")
    data = (
        subset(concat_folders(root, transform, lambda dirpath: dirpath.endswith(subfolder[0]))),
        subset(concat_folders(root, transform, lambda dirpath: dirpath.endswith(subfolder[1])))
    )
    return data


def extract_param(task, sub_task: Literal["sr", "dn", "jpeg", "compressed_sr"]):
    param = re.search(r'\d+', task.split("-")[1]).group()
    return float(param) if sub_task == "dn" else int(param)


def degradation(task: str) -> Optional[T.Compose]:
    lq_transform = []
    if 'compressed_sr' in task:
        return None
        # lq_transform.append(Jpeg(10))
        # lq_transform.append(Downsample(extract_param(task, "compressed_sr"), upsample_back=False))
        # return T.Compose(lq_transform)
    if 'sr' in task: lq_transform.append(Downsample(extract_param(task, "sr"), upsample_back=False))
    if 'jpeg' in task: lq_transform.append(Jpeg(extract_param(task, "jpeg")))
    if 'dn' in task: lq_transform.append(AddNoise(extract_param(task, "dn")))
    if 'denoising' in task:
        if 'blind' in task: lq_transform.append(AddNoise(50))
        else: lq_transform.append(AddNoise(float(task[-2:])))
    return T.Compose(lq_transform)


def paired_image_restoration_dataset(
        task: str,
        dataset_name: str,
        dataset_url: Optional[str] = None,
        subfolder: Optional[Union[str, Tuple[str, str]]] = None,
        split: Optional[Literal["train", "validation"]] = "train",
        resolution: Optional[int] = None,
        random_crop: bool = False,
        return_dict: bool = True,
        **unused_kwargs
) -> RestorationDataset:
    if "compressed_sr_swin2sr" in dataset_name:
        datasets = (
            dataset("DIV2K", split="validation", train_val_ratio=8 / 9)[0],
            dataset(
                "compressed_sr", split=None, subfolder="compressed_gt",
                dataset_url=f"https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_CompressedSR_X4_DIV2K_Valid.zip"
            )[0]
        )
    else:
        datasets = dataset(dataset_name, split, dataset_name.split("-")[0] in PAIRED_DATASETS, dataset_url, subfolder)
    return RestorationDataset(
        *datasets,
        degradation_func=degradation(task),
        resolution=resolution,
        random_crop=random_crop,
        return_dict=return_dict
    )


def datalaoder(task: str, dataset_name: Union[str, Tuple[str, ...]], max_samples: Optional[int] = None, **kwargs):
    if isinstance(dataset_name, str):
        dataset_name = (dataset_name,)

    with rank_zero_first():  # wrap with barriers to download on main proc. only
        d = ConcatRestorationDataset([
            paired_image_restoration_dataset(task, d, **kwargs)
            for d in dataset_name
        ])

    if max_samples and len(d) > max_samples:
        seed = datetime.now().hour  # fixed seed to ensure same split across processes
        generator = torch.Generator().manual_seed(seed)
        d, _ = random_split(d, [max_samples, len(d) - max_samples], generator=generator)

    batch_size = kwargs.get("batch_size", int(os.environ.get('batch_size', 10)))
    num_workers = kwargs.get("num_workers", int(os.environ.get('num_workers', 10)))
    resolution = d.dataset.resolution if isinstance(d, Subset) else d.resolution  # noqa
    # collate_fn = d.dataset.collate_fn if isinstance(d, Subset) else d.collate_fn  # noqa
    dl = DataLoader(d,
            batch_size=batch_size if resolution else 1,
            num_workers=num_workers if resolution else 0,
            collate_fn=None,
            shuffle=kwargs.get("shuffle", False)
        )
    return dl


if __name__ == '__main__':
    import torch
    from torchvision.utils import save_image

    ds = paired_image_restoration_dataset("sr-8_dn-25_jpeg-10", "Flickr2K", resolution=512, return_dict=False)
    input, target = ds[1]
    assert input.shape == target.shape == torch.Size([3, 512, 512])
    save_image(torch.stack((input, target)), "test.png")

    ds.resolution = 256  # changing resolution on-the-fly
    input, target = ds[1]
    assert input.shape == target.shape == torch.Size([3, 256, 256])
