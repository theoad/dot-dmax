from typing import Optional, Tuple, Literal

import torch
import torchvision.transforms as T

from dmax.data import dataset, Folder
from dmax.models import RestormerTask, Swin2SRTask, SwinIRTask, RELEASE_URL
from dmax.data.transforms import AddNoise, ToMultiple, Downsample


def restormer_results(task: RestormerTask, dataset_name: str, resolution: Optional[int] = None) -> Tuple[Folder, ...]:
    to_tensor = T.ToTensor()
    resize = T.Resize(resolution) if resolution else torch.nn.Identity()
    crop_to_8 = ToMultiple(8, mode="crop")

    if task == "gaussian_color_denoising_blind":
        if dataset_name not in ["CBSD68", "Kodak24", "McMaster", "Urban100"]:
            raise NotImplementedError()
        clean, = dataset(dataset_name.lower(), split=None, transform=T.Compose([to_tensor, resize, crop_to_8]))
        degraded, = dataset(dataset_name.lower(), split=None, transform=T.Compose([to_tensor, resize, crop_to_8, AddNoise(50)]))
        restored, = dataset(
            "restormer-gaussian_color_denoising_blind", split=None, subfolder=f"blind/{dataset_name}/50",
            transform=T.Compose([to_tensor, resize, crop_to_8]),
            dataset_url=f"1XfGMCAwo3ddgtwkxI0lm5aG-0FMY3RRT"
        )
    else:
        raise NotImplementedError()
    return clean, degraded, restored


def swin2sr_results(task: Swin2SRTask, dataset_name: str, resolution: Optional[int] = None) -> Tuple[Folder, ...]:
    release = RELEASE_URL['Swin2SR']
    to_tensor = T.ToTensor()
    resize = T.Resize(resolution) if resolution else torch.nn.Identity()
    crop_to_8 = ToMultiple(8, mode="crop")

    if task == "compressed_sr-4":
        clean, = dataset(
            "DIV2K", split="validation", train_val_ratio=8/9,
            transform=T.Compose([to_tensor, resize, crop_to_8])
        )
        degraded, restored = dataset(
            "swin2sr-compressed_sr-4-DIV2K", split=None, subfolder=("gt", "pred"),
            transform=T.Compose([to_tensor, resize, crop_to_8]),
            dataset_url=f"{release}/Swin2SR_CompressedSR_X4_DIV2K_Valid.zip"
        )
    elif task == "real_sr-4":
        clean = None
        degraded, restored = dataset(
            "swin2sr-real_sr-4", split=None, subfolder=("gt", "pred"),
            transform=T.Compose([to_tensor, resize, crop_to_8]),
            dataset_url=f"{release}/Swin2SR_RealworldSR_X4_RealSRSet.zip"
        )
    else:
        raise NotImplementedError()

    return clean, degraded, restored


def swinir_results(task: SwinIRTask, dataset_name: str, resolution: Optional[int] = None) -> Tuple[Folder, ...]:
    release = RELEASE_URL['SwinIR']
    to_tensor = T.ToTensor()
    resize = T.Resize(resolution) if resolution else torch.nn.Identity()
    crop_to_8 = ToMultiple(8, mode="crop")
    pad_to_8 = ToMultiple(8, mode="pad")

    if task == "classical_sr-4":
        if dataset_name not in ["B100", "Manga109", "Set5", "Set14", "Urban100"]:
            raise NotImplementedError()
        clean, = dataset(
            dataset_name.lower(), split=None,
            transform=T.Compose([to_tensor, resize, pad_to_8])
        )
        degraded, = dataset(
            dataset_name.lower(), split=None,
            transform=T.Compose([to_tensor, resize, pad_to_8, Downsample(4, True)])
        )
        restored, = dataset(
            "swinir-classical_sr-4-DIV2K", split=None, subfolder=dataset_name,
            transform=T.Compose([to_tensor, resize, crop_to_8]),
            dataset_url=f"{release}/visual_results_001_bicSR_DF2K_s64w8_SwinIR-M_x4.tar.gz"
        )
    elif task == "real_sr-4":
        clean = None
        degraded, restored = dataset(
            "swinir-real_sr-4", split=None, subfolder=("RealSRSet+5images", "SwinIR_BSRGAN"),
            transform=T.Compose([to_tensor, resize, crop_to_8]),
            dataset_url=f"{release}/visual_results_003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.tar.gz"
        )
    elif task == "color_dn-50":
        if dataset_name not in ["CBSD68", "Kodak24", "McMaster", "Urban100"]:
            raise NotImplementedError()
        clean, = dataset(
            dataset_name.lower(), split=None,
            transform=T.Compose([to_tensor, resize, crop_to_8])
        )
        degraded, = dataset(
            dataset_name.lower(), split=None,
            transform=T.Compose([to_tensor, resize, crop_to_8, AddNoise(50)])
        )
        restored, = dataset(
            "swinir-color_dn-50-DIV2K", split=None, subfolder=dataset_name,
            transform=T.Compose([to_tensor, resize, crop_to_8]),
            dataset_url=f"{release}/visual_results_005_colorDN_DFWB_s128w8_SwinIR-M_noise50.tar.gz"
        )
    else:
        raise NotImplementedError()
    return clean, degraded, restored


def ddrm_results(task: Literal["classical_sr_4_dn_25"], dataset_name: str, resolution: Optional[int] = None) -> Tuple[Folder, ...]:
    to_tensor = T.ToTensor()
    assert resolution == 512
    clean, = dataset(f"ddrm_{task}-restored", split=None, subfolder="original", transform=to_tensor)
    degraded, = dataset(f"ddrm_{task}-restored", split=None, subfolder="degraded", transform=to_tensor)
    restored, = dataset(f"ddrm_{task}-restored", split=None, subfolder="ddrm", transform=to_tensor)
    return clean, degraded, restored
