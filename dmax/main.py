import os
from os.path import join, exists
import warnings
import math
from typing import Tuple
from tqdm import tqdm
import jsonargparse
import numpy as np

from torchvision.utils import save_image
from torchvision.transforms.functional import resize

from accelerate import Accelerator
import lovely_tensors

import dmax.models as models
import dmax.data as data
from dmax.latent_w2 import LatentW2
from dmax.models.metrics import *


torch.set_float32_matmul_precision("highest")
lovely_tensors.monkey_patch()
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
warnings.filterwarnings('ignore', category=UserWarning, message='The default value of the antialias parameter')
warnings.filterwarnings('ignore', category=UserWarning, message='Arguments other than a weight')
warnings.filterwarnings('ignore', category=UserWarning, message="The parameter 'pretrained' is deprecated")
warnings.filterwarnings('ignore', category=UserWarning, message="torch.meshgrid")
warnings.filterwarnings('ignore', category=UserWarning, message="Metric `InceptionScore`")
warnings.filterwarnings('ignore', category=UserWarning, message="Metric `Kernel Inception Distance`")
warnings.filterwarnings('ignore', category=FutureWarning, message="Accessing config attribute `block_out_channels`")


@torch.inference_mode()
def latent_dmax(
        model: models.Model,
        task: models.Task,
        exp_name: str = "defaults",
        vae: Optional[str] = "stabilityai/sd-vae-ft-ema",  # if `None`, will work in the pixel space
        natural_image_set: Tuple[str, ...] = ("imagenet-train",),
        degraded_set: Optional[Tuple[str, ...]] = ("imagenet-train",),  # zero shot if left `None`
        quantitative_set: Tuple[str, ...] = ("imagenet-val",),
        qualitative_set: Tuple[str, ...] = ("set5","set14","b100"),
        block_size: int = 5,
        kernel_size: Optional[int] = None,
        sigma: Optional[float] = None,
        shared: bool = True,
        unpaired: bool = True,
        sensitivity: float = 1e-7,  # increase if your model is far away from nat. images.
        store_features: bool = False,
        train_resolution: Optional[int] = 512,
        test_resolution: Optional[int] = None,  # use original image size if left `None`
        max_clean_samples: int = 16,
        max_deg_samples: int = 16,
        max_quantitative_samples: int = int(1e6),
        pd_tradeoff_pts: Tuple[float, ...] = tuple((round(v,1) for v in np.linspace(-1,2,31))),
        latent_interp: bool = False,
        pbar: bool = True
):
    """
    Applies the Deep Optimal Transport (DOT) for enhanced image restoration
    Args:
        model: The restoration model to enhance.
        task: The restoration task.
        exp_name: The experiment name (results will be located under output/`task`/`exp_name`.
        vae: The VAE release to use.
        natural_image_set: Dataset(s) of images from which target distribution parameters are computed.
        degraded_set: Dataset(s) of images which will be degraded, then restored and will serve.
        to compute the source distribution parameters.
        quantitative_set: Dataset(s) on which to compute quantitative performance.
        qualitative_set: Dataset(s) of images which are saved for qualitative evaluation.
        block_size: The patch-size `p` in latent space. All overlapping patches of size (c,p,p) will be transported.
        kernel_size: The size of the wieghting kernel used when averaging back the transported overlapping patches.
        sigma: The standard deviation of the weighting kernel `folding_kernel_size`.
        shared: Whether or not the have a different transport operator for each patch.
        unpaired: If ``False`` will use the same images for estimatiing source and target distributions.
        sensitivity: Small positive value for numerical stability.
        store_features: If ``True`` will store in memory the latent features instead of computing the paramters on-the-fly.
        train_resolution: When computing distribution parameters, will resize images to `(train_resolution, train_resolution)`.
        test_resolution:  When enhancing images from the qualitative set, will resize to `(test_resolution, test_resolution)` if given.
        max_clean_samples: Max. number of natural images from which target distribution parameters are computed.
        max_deg_samples: Max. number of restored images from which source distribution parameters are computed.
        max_quantitative_samples: Max. number of paires on which to compute the evaluation performance.
        pd_tradeoff_pts: The values of \alpha on which to compute the evaluation performance.
        latent_interp: If ``True`` will apply eq. (4) from the paper directly in the latent space.
        pbar: If ``False`` will disable progress bar.
    """
    def dmax_field(alpha: float): return f"x{str(alpha).replace('.', '').replace('-', '')}" + ("n" if alpha < 0 else "")

    # we cache the dmax object to avoid computing the real statistics twice
    output = join("output", model.lower(), task.lower() + "_res-" + str(train_resolution), exp_name)
    dmax_cache = join(output, 'dmax.pth')
    os.makedirs(output, exist_ok=True)

    # use hugging-face's accelerate library to handle distributed env.
    accelerator = Accelerator()

    # models and metrics
    kernel_size = kernel_size or block_size
    kernel_size = min(kernel_size, block_size)
    sigma = sigma or float('inf')

    device = accelerator.device
    pbar = pbar and accelerator.is_local_main_process
    dl_kwargs = dict(resolution=train_resolution, return_dict=False)
    kid_subset = 100 if "compressed_sr_swin2sr" in quantitative_set else 1000

    model = getattr(models, model.lower())(task, pretrained=True).to(device)
    dmax = LatentW2(
        vae, block_size, kernel_size, sigma,
        shared=shared, resolution=train_resolution, eps=sensitivity, store_features=store_features
    ).to(device)
    metrics = torch.nn.ModuleDict(
        {k: MetricCollection([PSNR(), SSIM(), LPIPS(), FID(), IS(), KID(subset_size=kid_subset)])
         for k in ["x", "xe", "y", "x*"] + [dmax_field(alpha) for alpha in pd_tradeoff_pts]}
    ).to(device)

    if exists(dmax_cache):
        accelerator.print(f"found cached dmax state at {dmax_cache}")
        dmax.load_state_dict(torch.load(dmax_cache, map_location=accelerator.device), strict=False)
        dmax._computed = None
    else:
        # compute latent target statistics with natural images
        target_dl = data.datalaoder(task, natural_image_set, max_clean_samples, split="train", **dl_kwargs)
        target_dl = accelerator.prepare_data_loader(target_dl)
        accelerator.wait_for_everyone()

        for y, x in tqdm(target_dl, f"Computing {natural_image_set} target latent parameters", disable=not pbar):
            dmax.update(x, distribution="target")
            if not unpaired: dmax.update(model(y), distribution="source")

        if degraded_set and unpaired:
            # compute latent source statistics with degraded images
            source_dl = data.datalaoder(task, degraded_set, max_deg_samples, split="train", **dl_kwargs)
            source_dl = accelerator.prepare_data_loader(source_dl)
            accelerator.wait_for_everyone()
            for y, x in tqdm(source_dl, f"Computing {natural_image_set} source latent parameters", disable=not pbar):
                dmax.update(model(y), distribution="source")

        dmax.compute()

        accelerator.wait_for_everyone()
        accelerator.save(dmax.state_dict(), dmax_cache)
        # accelerator.print(f"saved dmax states at {dmax_cache}")

    dmax.compute()
    accelerator.print(f"{' ' if vae is not None else 'Latent '}w2 distance: {dmax.compute().cpu().item():.2f}")

    # compute quantitative performance on the quantitative image set
    val_dl = data.datalaoder(task, quantitative_set, max_quantitative_samples, split=None, **dl_kwargs)
    val_dl = accelerator.prepare_data_loader(val_dl)
    for y, x in tqdm(val_dl, f"Computing {quantitative_set} performance", disable=not pbar):
        x_star = model(y)
        metrics["x"].update(x, x)
        metrics["xe"].update(dmax.decode(dmax.encode(x)).clip(0, 1), x)
        metrics["y"].update(resize(y, x.shape[-2:]).clip(0, 1), x)
        metrics["x*"].update(x_star.clip(0, 1), x)

        x_hat0 = dmax.transport(x_star, zero_shot=degraded_set is None, pg_star=0.)
        for alpha in pd_tradeoff_pts:  # traversing PD tradeoff
            if latent_interp:
                x_hat0_alpha = dmax.transport(x_star, zero_shot=degraded_set is None, pg_star=alpha).clip(0, 1)
            else:
                x_hat0_alpha = (x_hat0 * (1 - alpha) + x_star * alpha).clip(0, 1)
            metrics[dmax_field(alpha)].update(x_hat0_alpha, x)

    for metric in metrics.values():
        metric.compute()

    if accelerator.is_local_main_process:
        print_results(metrics, join(output, f"{'-'.join(quantitative_set)}-{train_resolution}.txt"))
        print_results(metrics)

    # testing  results
    dl_kwargs = dict(resolution=test_resolution if shared else train_resolution, random_crop=False, return_dict=False)
    test_dl = data.datalaoder(task, qualitative_set, split=None, **dl_kwargs)
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        i = 0; zero_fill = int(math.log10(len(test_dl))) + 1
        for y, x in tqdm(test_dl, f"Saving {qualitative_set} results", disable=not pbar):
            x = x.to(device); y = y.to(device)
            x_star = model(y)
            x_star = x_star[..., :x.size(-2), :x.size(-1)]
            x = x[..., :x_star.size(-2), :x_star.size(-1)]
            x_hat0 = dmax.transport(x_star, zero_shot=degraded_set is None, pg_star=0.)
            x_hat0 = x_hat0[..., :x.size(-2), :x.size(-1)]
            x_hat05 = x_hat0 * 0.5 + x_star * 0.5
            xe = dmax.decode(dmax.encode(x))
            collage = (resize(y, x.shape[-2:]), x_star, x_hat0, x)

            out_path = join(output, str(i).zfill(zero_fill)); os.makedirs(out_path, exist_ok=True)
            save_image(x, join(out_path, 'x.png'))
            save_image(y, join(out_path, 'y.png'))
            save_image(x_star, join(out_path, 'x_star.png'))
            save_image(x_hat0, join(out_path, 'x_hat00.png'))
            save_image(x_hat05, join(out_path, 'x_hat05.png'))
            save_image(xe, join(out_path, 'xe.png'))
            save_image(torch.cat(collage, dim=-1), join(out_path, 'collage.png'), nrow=1, padding=0,)
            i += 1


if __name__ == '__main__':
    jsonargparse.CLI()
