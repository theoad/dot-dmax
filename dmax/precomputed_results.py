import os
from tqdm import tqdm
from math import log10
import jsonargparse
import warnings
import numpy as np

import torch.distributed
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from accelerate import Accelerator

import dmax.data.results as results
from dmax.latent_w2 import LatentW2
from dmax.models.metrics import *
from utils.ddp import rank_zero_first


warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
warnings.filterwarnings('ignore', category=UserWarning, message='The default value of the antialias parameter')
warnings.filterwarnings('ignore', category=UserWarning, message='Arguments other than a weight')
warnings.filterwarnings('ignore', category=UserWarning, message="The parameter 'pretrained' is deprecated")
warnings.filterwarnings('ignore', category=UserWarning, message="torch.meshgrid")
warnings.filterwarnings('ignore', category=UserWarning, message="Metric `InceptionScore`")
warnings.filterwarnings('ignore', category=UserWarning, message="Metric `Kernel Inception Distance`")
warnings.filterwarnings('ignore', category=FutureWarning, message="Accessing config attribute `block_out_channels`")


@torch.inference_mode()
def transport(model: str, task: str, data: str, resolution: Optional[int] = 512):
    def dmax_field(alpha: float): return f"x{str(alpha).replace('.', '').replace('-', '')}" + ("n" if alpha < 0 else "")
    pd_tradeoff_pts = tuple((round(v,1) for v in np.linspace(1,0,11)))

    accelerator = Accelerator()
    device = accelerator.device

    output = os.path.join("results", '-'.join((model.lower(), task.lower(), data.lower(), str(resolution))))
    os.makedirs(output, exist_ok=True)

    with rank_zero_first():
        clean, degraded, restored = getattr(results, model.lower() + "_results")(task, data, resolution)

    dmax = LatentW2(block_size=5, folding_kernel_size=5, eps=1e-7, vae="stabilityai/sd-vae-ft-ema").to(device)
    metrics = torch.nn.ModuleDict({
        k: MetricCollection([PSNR(), SSIM(), LPIPS(), FID(), IS(), KID()])
        for k in ["x", "y", "x*", "xhat0"] + [dmax_field(alpha) for alpha in pd_tradeoff_pts]
    }).to(device)

    r_dl, d_dl, c_dl = map(DataLoader, (restored, degraded, clean))
    r_dl, d_dl, c_dl = accelerator.prepare(r_dl, d_dl, c_dl)
    i = 0
    for x_star, x in tqdm(zip(r_dl, c_dl), desc="training dmax", disable=not accelerator.is_local_main_process):
        if i%2: dmax.update(x_star, "source")
        else: dmax.update(x, "target")
        i+=1
        if i//2 * accelerator.num_processes >= 100:
            break

    i = 0
    for x_star, y, x in tqdm(zip(r_dl, d_dl, c_dl), desc="computing performance", disable=not accelerator.is_local_main_process):
        x_star = x_star[..., :x.size(-2), :x.size(-1)]
        x = x[..., :x_star.size(-2), :x_star.size(-1)]
        y = y[..., :x_star.size(-2), :x_star.size(-1)]
        xhat0 = dmax.transport(x_star).clip(0, 1)
        metrics["x"].update(x, x)
        metrics["y"].update(y, x)
        metrics["x*"].update(x_star, x)
        metrics["xhat0"].update(xhat0, x)

        out_path = os.path.join(output, str(i).zfill(int(log10(len(c_dl)))+1))
        os.makedirs(out_path, exist_ok=True)
        collage = (y, x_star)

        for alpha in pd_tradeoff_pts:  # traversing PD tradeoff
            x_hat0_alpha = (xhat0 * (1 - alpha) + x_star * alpha).clip(0, 1)
            metrics[dmax_field(alpha)].update(x_hat0_alpha, x)
            save_image(x_hat0_alpha, os.path.join(out_path, f'{dmax_field(alpha)}.png'))
            collage += (x_hat0_alpha,)

        collage += (x,)
        save_image(x, os.path.join(out_path, 'x.png'))
        save_image(y, os.path.join(out_path, 'y.png'))
        save_image(x_star, os.path.join(out_path, 'x_star.png'))
        save_image(xhat0, os.path.join(out_path, 'xhat0.png'))
        save_image(torch.cat(collage, dim=-1), os.path.join(out_path, 'collage.png'), nrow=1, padding=0,)
        i += 1

    for metric in metrics.values():
        metric.compute()

    if accelerator.is_local_main_process:
        print_results(metrics, os.path.join(output, "results.txt"))
        print_results(metrics)


if __name__ == '__main__':
    jsonargparse.CLI()
