

# DOT-Dmax
Official Pytorch implementation of _Deep Optimal Transport: A Practical Algorithm for Photo-realistic Image Restoration_

> We propose an image restoration algorithm that can control the perceptual quality and/or the mean square error (MSE) of any pre-trained model, trading one over the other at test time. Our algorithm is few-shot: Given about a dozen images restored by the model, it can significantly improve the perceptual quality and/or the MSE of the model for newly restored images without further training.

![Collage](assets/teaser.png)

## Dependencies
We used [miniconda3](https://docs.conda.io/en/latest/miniconda.html) and [pip3](https://pip.pypa.io/en/stable/) to manage dependencies
```bash
conda create -n dmax python=3.8
conda activate dmax
pip install -e .
cd dmax
```

## Usage
Example of our algorithm applied to the [SwinIR](https://github.com/JingyunLiang/SwinIR) model for SISRx4.
```python
from PIL import Image

import torch
from torchvision.transforms.functional import to_tensor, resize
from torchvision.utils import save_image
from datasets import load_dataset

from dmax.latent_w2 import LatentW2
from dmax.models import swinir

device = "cuda" if torch.cuda.is_available() else "cpu"
dmax = LatentW2("stabilityai/sd-vae-ft-ema").to(device)
model = swinir("classical_sr-4", pretrained=True).to(device)
dataset = iter(load_dataset("imagenet-1k", split="train", streaming=True))

for _ in range(100): # arbitrary resolution & aspect ratios
    x = to_tensor(next(dataset)['image']).to(device).unsqueeze(0)
    dmax.update(x, distribution="target")  # update nat. image statistics

for _ in range(100):  # unpaired updates
    x = to_tensor(next(dataset)['image']).to(device).unsqueeze(0)
    y = resize(x, (x.size(-2)//4, x.size(-1)//4), antialias=True)  # degrade image
    x_star = model(y)  # restore with the pre-trained model
    dmax.update(x_star, distribution="source")  # update model statistics

w2 = dmax.compute()  # compute the latent transport operator & W2 distance

x = to_tensor(Image.open("../assets/baboon.png")).to(device).unsqueeze(0) 
y = resize(x, (x.size(-2)//4, x.size(-1)//4), antialias=True)
x_star = model(y) 
xhat_0 = dmax.transport(x_star) # enhance new images

collage = torch.cat([resize(y, x.shape[-2:]), x_star, xhat_0, x], dim=-1).to(device)
save_image(collage, "demo.png", nrow=1, padding=0)
```

## Paper Results
### PyDrive-API
Our algorithm enhances existing methods (we tested [SwinIR](https://github.com/JingyunLiang/SwinIR), [Swin2SR](https://github.com/mv-lab/swin2sr), [Restormer](https://github.com/swz30/Restormer), [ESRGAN](https://github.com/xinntao/ESRGAN) and [DDRM](https://github.com/bahjat-kawar/ddrm)).
Instead of imposing on users to download manually third party code, data or weights, we automate everything using google drive's API.

Note: The following only applies to *your* script, and does not give access to other users.
Nevertheless, we recommend revoking the script's access after the download is complete.

- Follow the steps of [google's pydrive quickstart](https://developers.google.com/drive/api/v3/quickstart/python) and place your `credentials.json` under the `dot-dmax` repository.
- Run `python data/gdrive.py init` (must be on a local machine, connected to a display). If the warning _"Google hasn’t verified this app"_ occurs, click `advance` and then `Go to Local (unsafe)`.


### Hardware Setup
We abstract hardware dependency using hugging-face's [accelerate](https://huggingface.co/docs/accelerate/index) library.
Configure your environment before launching the scripts by running
```bash
accelerate config
```
Because we compute evaluation for many values of $\alpha$, we distribute evaluation across 8 A6000 GPUs with `batch_size=10`.
Reduce this value if you encounter any CUDA out-of-memory issues using
```bash
export batch_size=8
```
After configuring your hardware, launch distributed jobs by replacing `python main.py <args>`  by `accelerate launch main.py <args>`
### Command-line
```bash
python main.py --help
```
```bash
usage: main.py [-h] [--config CONFIG] [--print_config[=flags]] [--exp_name EXP_NAME] [--vae VAE] [--natural_image_set [ITEM,...]] [--degraded_set DEGRADED_SET] [--quantitative_set [ITEM,...]] [--qualitative_set [ITEM,...]] [--block_size BLOCK_SIZE] [--kernel_size KERNEL_SIZE] [--sigma SIGMA]
               [--shared {true,false}] [--unpaired {true,false}] [--sensitivity SENSITIVITY] [--store_features {true,false}] [--train_resolution TRAIN_RESOLUTION] [--test_resolution TEST_RESOLUTION] [--max_clean_samples MAX_CLEAN_SAMPLES] [--max_deg_samples MAX_DEG_SAMPLES]
               [--max_quantitative_samples MAX_QUANTITATIVE_SAMPLES] [--pd_tradeoff_pts [ITEM,...]] [--latent_interp {true,false}] [--pbar {true,false}]
               {Restormer,Swin2SR,SwinIR,ESRGAN,ESRGAN_PSNR,NLM} task

Applies the Deep Optimal Transport (DOT) for enhanced image restoration

positional arguments:
  {Restormer,Swin2SR,SwinIR,ESRGAN,ESRGAN_PSNR,NLM}
                        (required, type: Literal['Restormer', 'Swin2SR', 'SwinIR', 'ESRGAN', 'ESRGAN_PSNR', 'NLM'])
  task                  (required, type: Union[Literal['deraining', 'real_denoising', 'motion_deblurring', 'single_image_defocus_deblurring', 'gaussian_color_denoising_blind', 'gaussian_color_denoising_sigma15', 'gaussian_color_denoising_sigma25', 'gaussian_color_denoising_sigma50'], Literal['classical_sr-2',
                        'classical_sr-4', 'lightweight_sr-2', 'real_sr-4', 'compressed_sr-4', 'color_jpeg_car-10', 'color_jpeg_car-20', 'color_jpeg_car-30', 'color_jpeg_car-40'], Literal['classical_sr-2', 'classical_sr-3', 'classical_sr-4', 'classical_sr-8', 'lightweight_sr-2', 'lightweight_sr-3',
                        'lightweight_sr-4', 'real_sr-2', 'real_sr-4', 'jpeg_car-10', 'jpeg_car-20', 'jpeg_car-30', 'jpeg_car-40', 'color_jpeg_car-10', 'color_jpeg_car-20', 'color_jpeg_car-30', 'color_jpeg_car-40', 'gray_dn-15', 'gray_dn-25', 'gray_dn-50', 'color_dn-15', 'color_dn-25', 'color_dn-50']])

optional arguments:
  -h, --help            Show this help message and exit.
  --config CONFIG       Path to a configuration file.
  --print_config[=flags]
                        Print the configuration after applying all other arguments and exit. The optional flags customizes the output and are one or more keywords separated by comma. The supported flags are: comments, skip_default, skip_null.
  --exp_name EXP_NAME   (type: str, default: defaults)
  --vae VAE             (type: Union[str, null], default: stabilityai/sd-vae-ft-ema)
  --natural_image_set [ITEM,...]
                        (type: Tuple[str, ], default: ('imagenet-train',))
  --degraded_set DEGRADED_SET
                        (type: Union[Tuple[str, ], null], default: ('imagenet-train',))
  --quantitative_set [ITEM,...]
                        (type: Tuple[str, ], default: ('imagenet-val',))
  --qualitative_set [ITEM,...]
                        (type: Tuple[str, ], default: ('set5', 'set14', 'b100'))
  --block_size BLOCK_SIZE
                        (type: int, default: 5)
  --kernel_size KERNEL_SIZE
                        (type: Union[int, null], default: null)
  --sigma SIGMA         (type: Union[float, null], default: null)
  --shared {true,false}
                        (type: bool, default: True)
  --unpaired {true,false}
                        (type: bool, default: True)
  --sensitivity SENSITIVITY
                        (type: float, default: 1e-07)
  --store_features {true,false}
                        (type: bool, default: False)
  --train_resolution TRAIN_RESOLUTION
                        (type: Union[int, null], default: 512)
  --test_resolution TEST_RESOLUTION
                        (type: Union[int, null], default: null)
  --max_clean_samples MAX_CLEAN_SAMPLES
                        (type: int, default: 16)
  --max_deg_samples MAX_DEG_SAMPLES
                        (type: int, default: 16)
  --max_quantitative_samples MAX_QUANTITATIVE_SAMPLES
                        (type: int, default: 1000000)
  --pd_tradeoff_pts [ITEM,...]
                        (type: Tuple[float, ], default: (-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0))
  --latent_interp {true,false}
                        (type: bool, default: False)
  --pbar {true,false}   (type: bool, default: True)
```

### Reproducing results
```bash
python main.py ESRGAN classical_sr-4                        # ESRGAN    (SISRx4)
python main.py SwinIR classical_sr-4                        # SwinIR    (SISRx4)
python main.py SwinIR jpeg_car-10                           # SwinIR    (JPEGq10)
python main.py Restormer gaussian_color_denoising_sigma50   # Restormer (AWGNs50)
python main.py nlm color_dn-50                              # NLM       (AWGNs50)

python main.py Swin2SR compressed_sr-4 \                    # Swin2SR   (SISRx4 + JPEGq10)
 --natural_image_set ["compressed_sr_swin2sr"] \
 --degraded_set ["compressed_sr_swin2sr"] \
 --quantitative_set ["compressed_sr_swin2sr"] \
 --qualitative_set ["compressed_sr_swin2sr"]
```