from typing import Literal, Union
import os
import torch
import requests

from dmax.utils import CACHE
import dmax.models.model_registery as registery
from dmax.utils.drive import download_file, build_service
from dmax.utils.ddp import rank_zero_first


MODEL_CACHE = os.path.join(CACHE, "models")
os.makedirs(MODEL_CACHE, exist_ok=True)


RELEASE_URL = dict(
    Restormer="https://github.com/swz30/Restormer/releases/download/v1.0",
    Swin2SR="https://github.com/mv-lab/swin2sr/releases/download/v0.0.1",
    SwinIR="https://github.com/JingyunLiang/SwinIR/releases/download/v0.0",
    ESRGAN="1TPrz5QKd8DHHt1k8SRtm6tMiPjz_Qene",
    ESRGAN_PSNR="1pJ_T-V1dpb1ewoEra1TGSWl5e6H7M4NN",
)

# supported degradations from Restormer
RestormerTask = Literal[
    "deraining",
    "real_denoising",
    "motion_deblurring",
    "single_image_defocus_deblurring",
    "gaussian_color_denoising_blind",
    "gaussian_color_denoising_sigma15",
    "gaussian_color_denoising_sigma25",
    "gaussian_color_denoising_sigma50",
]

# supported degradations from Swin2SR
Swin2SRTask = Literal[
    "classical_sr-2", "classical_sr-4",
    "lightweight_sr-2",
    "real_sr-4",
    "compressed_sr-4",
    "color_jpeg_car-10", "color_jpeg_car-20", "color_jpeg_car-30", "color_jpeg_car-40",
]

# supported degradations from SwinIR
SwinIRTask = Literal[
    "classical_sr-2", "classical_sr-3", "classical_sr-4", "classical_sr-8",
    "lightweight_sr-2", "lightweight_sr-3", "lightweight_sr-4",
    "real_sr-2", "real_sr-4",
    "jpeg_car-10", "jpeg_car-20", "jpeg_car-30", "jpeg_car-40",
    "color_jpeg_car-10", "color_jpeg_car-20", "color_jpeg_car-30", "color_jpeg_car-40",
    "gray_dn-15", "gray_dn-25", "gray_dn-50",
    "color_dn-15", "color_dn-25", "color_dn-50"
]

Task = Union[RestormerTask, Swin2SRTask, SwinIRTask]
Model = Literal["Restormer", "Swin2SR", "SwinIR", "ESRGAN", "NLM", "Identity"]


def weights(release_url: str, release_name: str, state_key: str = 'params', verb: bool = False):
    model_path = os.path.join(MODEL_CACHE, release_name)
    if os.path.exists(model_path):
        if verb:
            print(f"found cached checkpoint at {model_path}")
    else:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        if release_url.startswith("https://"):
            r = requests.get(f"{release_url}/{release_name}", allow_redirects=True)
            open(model_path, 'wb').write(r.content)
        else:
            service = build_service()
            download_file(service, model_path, release_url)

    state = torch.load(model_path)
    return state[state_key] if state_key and (state_key in state.keys()) else state


def config_and_release(task: Task, model_type: Model, large_model: bool = True):
    if model_type == "NLM":
        assert 'dn' in task, "need to specify noise standard deviation for `NLM`"
        noise_sigma = float(task.split('-')[-1])
        config = {'sigma': noise_sigma / 255.,'patch_size': 5, 'patch_distance': 32}
        release_name = None

    elif "ESRGAN" in model_type:
        assert task == "classical_sr-4"
        config = {'in_nc': 3, 'out_nc': 3, 'nf': 64, 'nb': 23, 'gc': 32}
        release_name = f"RRDB_{model_type}_x4.pth"

    elif model_type == "Restormer":
        assert task in RestormerTask.__args__  # noqa
        release_name = f"restormer_{task}.pth"
        config = {
            'inp_channels': 3, 'out_channels': 3, 'dim': 48, 'num_blocks': [4, 6, 6, 8], 'num_refinement_blocks': 4,
            'heads': [1, 2, 4, 8], 'ffn_expansion_factor': 2.66, 'bias': False, 'dual_pixel_task': False,
            'LayerNorm_type': 'BiasFree' if 'denoising' in task else 'WithBias'
        }

    elif model_type in ["SwinIR", "Swin2SR"]:
        task_name = task.split('-')[0]
        scale = int(task.split('-')[-1]) if 'sr' in task else 1
        noise_sigma = float(task.split('-')[-1]) if 'dn' in task else 1
        jpeg_quality = int(task.split('-')[-1]) if 'jpeg' in task else 1

        config = {
            'upscale': scale, 'in_chans': 3, 'img_size': 64, 'window_size': 8, 'img_range': 1., 'num_heads': [6,] * 6,
            'depths': [6,] * 6, 'embed_dim': 180, 'mlp_ratio': 2, 'upsampler': 'pixelshuffle' if scale > 1 else '',
            'resi_connection': '1conv'
        }

        # 001 classical image sr
        if task_name == 'classical_sr':
            available_scales = [2, 3, 4, 8] if model_type == "SwinIR" else [2, 4]
            if scale not in available_scales: raise ValueError(
                f"for `{model_type}` model in `{task_name}` only `scale = {available_scales}` are supported."
            )
            release_name = f"001_classicalSR_DF2K_s64w8_SwinIR-M_x{scale}.pth" if model_type == "SwinIR" else \
                           f"Swin2SR_ClassicalSR_X{scale}_64.pth"

        # 002 lightweight image sr
        # use 'pixelshuffledirect' to save parameters
        elif task_name == 'lightweight_sr':
            available_scales = [2, 3, 4] if model_type == "SwinIR" else [2, ]
            if scale not in available_scales: raise ValueError(
                f"for `{model_type}` model in `{task_name}` only `scale = {available_scales}` are supported."
            )
            config['depths'] = config['num_heads'] = [6,] * 4
            config['embed_dim'] = 60
            config['upsampler'] = 'pixelshuffledirect'
            release_name = f"002_lightweightSR_DIV2K_s64w8_SwinIR-S_x{scale}.pth" if model_type == "SwinIR" else \
                           f"Swin2SR_Lightweight_X{scale}_64.pth"

        # 002.5 compressed image sr
        # use training_patch_size = 48 to save memory
        elif task_name == "compressed_sr":
            available_scales = [] if model_type == "SwinIR" else [4, ]
            if scale not in available_scales: raise ValueError(
                f"for `{model_type}` model in `{task_name}` only `scale = {available_scales}` are supported."
            )
            config['img_size'] = 48
            config['upsampler'] = 'pixelshuffle_aux'
            release_name = f"Swin2SR_CompressedSR_X{scale}_48.pth"

        # 003 real-world image sr
        elif task_name == 'real_sr':
            # use 'nearest+conv' to avoid block artifacts
            config['upsampler'] = 'nearest+conv'
            available_scales = [2, 4] if model_type == "SwinIR" else ([4, ] if not large_model else [])
            if scale not in available_scales: raise ValueError(
                f"for `{model_type}` model in `{task_name}` only `scale = {available_scales}` are supported."
            )
            if not large_model:
                release_name = f"003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x{scale}_PSNR.pth" if model_type == "SwinIR" else \
                               f"Swin2SR_RealworldSR_X{scale}_64_BSRGAN_PSNR.pth"
            else:
                # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
                config['depths'] = [6,] * 8
                config['num_heads'] = [8,] * 8
                config['embed_dim'] = 240
                config['resi_connection'] = '3conv'
                release_name = f"003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x{scale}_PSNR.pth"

        # 004 & 005 image denoising
        elif 'dn' in task_name:
            available_sigma = [15., 25., 50.] if model_type == "SwinIR" else []
            if noise_sigma not in available_sigma: raise ValueError(
                f"for `{model_type}` model in `{task_name}` only `sigma = {available_sigma}` are supported."
            )
            config['in_chans'] = 1  if 'gray' in task_name else 3
            config['img_size'] = 128
            release_name = f"004_grayDN_DFWB_s128w8_SwinIR-M_noise{noise_sigma}.pth" if 'grey' in task_name else \
                           f"005_colorDN_DFWB_s128w8_SwinIR-M_noise{noise_sigma}.pth"

        # 006 grayscale JPEG compression artifact reduction
        # use window_size=7 because JPEG encoding uses 8x8; use img_range=255 because it's sligtly better than 1
        elif 'jpeg' in task_name:
            available_qualities = [10, 20, 30, 40] if model_type == "SwinIR" else []
            if jpeg_quality not in available_qualities: raise ValueError(
                f"for `{model_type}` model in `{task_name}` only `qualities = {available_qualities}` are supported."
            )
            config['in_chans'] = 3 if 'color' in task_name else 1
            config['img_size'] = 126
            config['window_size'] = 7
            config['img_range'] = 255
            release_name = f"006_{'color' if 'color' in task_name else ''}CAR_DFWB_s126w7_SwinIR-M_jpeg{jpeg_quality}.pth" \
            if model_type == "SwinIR" else "Swin2SR_Jpeg_dynamic.pth"
        else:
            raise NotImplementedError(f"task `{task}` not supported.")
    else:
        raise NotImplementedError(f"Model `{model_type}` not supported")
    return config, release_name


def model(task: Task, model_type: Model, large_model: bool = False, pretrained: bool = True):
    config, release_name = config_and_release(task, model_type, large_model=large_model)
    model_cls = getattr(registery, model_type)
    _model = model_cls(**config)
    if release_name is not None and pretrained:
        release_url = RELEASE_URL[model_type]
        params_key = 'params_ema' if 'real_sr' in task else 'params'
        with rank_zero_first():
            pretrained_weights = weights(release_url, release_name, state_key=params_key)
        _model.load_state_dict(pretrained_weights)
        _model.eval()
    return _model


def restormer(task: RestormerTask, pretrained: bool = True) -> registery.Restormer:
    return model(task, "Restormer", pretrained=pretrained)


def swinir(task: SwinIRTask, pretrained: bool = True) -> registery.SwinIR:
    return model(task, "SwinIR", pretrained=pretrained)


def swin2sr(task: Swin2SRTask, pretrained: bool = True) -> registery.Swin2SR:
    return model(task, "Swin2SR", pretrained=pretrained)


def esrgan(task: RestormerTask, pretrained: bool = True) -> registery.ESRGAN:
    return model(task, "ESRGAN", pretrained=pretrained)


def nlm(task: Task, pretrained: bool = True) -> registery.NLM:
    return model(task, "NLM", pretrained=pretrained)


def identity(task: Task, pretrained: bool = True):
    return torch.nn.Identity()
