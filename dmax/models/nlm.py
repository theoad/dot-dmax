import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dmax.utils.tile import Tile


def nlm(image: Tensor, sigma: float, patch_size: int = 5, H: float = 1.) -> torch.Tensor:
    b, c, h, w = image.shape
    patches = F.unfold(F.pad(image, (patch_size // 2,) * 4, mode="replicate"), patch_size)  # B, CxPxP, HxW
    # Get the noisy pixel and its neighborhood
    y = image.flatten(2)  # B, C, HxW
    yN = patches.unflatten(1, (c, patch_size ** 2))  # B, C, PxP, HxW
    yN = torch.cat([yN[:, :, :patch_size ** 2 // 2], yN[:, :, patch_size ** 2 // 2 + 1:]], dim=2)
    yN = yN.flatten(1, 2).transpose(1, 2)  # B, C, PxP-1, HxW --> B, CxPxP-C, HxW --> B, HxW, CxPxP-C
    dist = torch.cdist(yN.double(), yN.double())
    dist += torch.diag_embed(torch.ones_like(dist[..., 0]) * float('inf'))
    density = torch.exp(- dist / (2 * H ** 2 * sigma ** 2))  # B, HxW, HxW
    weight = density / density.sum(-1, keepdim=True)  # B, HxW, HxW
    mmse = weight @ y.double().transpose(1, 2)  # B, HxW, C
    return mmse.float().transpose(1, 2).unflatten(-1, (h, w))  # B, C, H, W

    # b,c,h,w = image.shape
    # # Get all the overlapping patches
    # patches = F.unfold(F.pad(image, (self.patch_size//2,) * 4, mode="replicate"), self.patch_size)  # B, CxPxP, HxW
    #
    # # Remove the noisy pixel from its neighborhood before computing similarities
    # yN = patches[:, self.patch_size ** 2//2::self.patch_size ** 2].zero_() # B, CxPxP, HxW
    # yN = yN.transpose(1, 2)  # B, HxW, CxPxP
    # dist = torch.cdist(yN, yN)  # B, HxW, HxW
    #
    # # Remove self-similarity - we are interested in the similarity with the other patches in the images
    # dist += torch.diag_embed(torch.ones_like(dist[..., 0]) * float('inf'))
    #
    # # Apply Gaussian p.d.f and normalize
    # density = torch.exp(-(dist.double() / (2 * self.h * self.sigma)) ** 2)  # noqa B, HxW, HxW
    # weight = density / density.sum(-1, keepdim=True)
    #
    # # The denoised pixel is the weighted average of the pixels in the image
    # mmse = image.double().flatten(-2) @ weight.transpose(1, 2)  # B, C, HxW
    # return mmse.unflatten(-1, (h, w)).float()  # B, C, H, W


class NLM(nn.Module):
    def __init__(self, sigma: float, patch_size: int = 5, patch_distance: int = 32) -> None:
        super().__init__()
        self.h = (patch_size // 2) ** 0.5
        self.sigma = sigma
        self.patch_size = patch_size
        self.patch_distance = patch_distance

    def forward(self, image: Tensor) -> Tensor:
        with Tile(image, self.patch_distance, overlap=1/4) as T:
            return T.fold_back(nlm(T.tiles, self.sigma, self.patch_size, self.h))

    def extra_repr(self) -> str:
        return f"sigma={self.sigma:.2f}, patch_size={self.patch_size}, patch_distance={self.patch_distance}"


if __name__ == '__main__':
    from PIL import Image
    import requests

    from torchvision.transforms.functional import to_tensor
    from torchvision.utils import save_image

    from torchmetrics.image import PeakSignalNoiseRatio

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    sigma_255 = 50
    sigma = sigma_255 / 255
    mmse = NLM(sigma, patch_size=5, patch_distance=32).to(device)
    psnr = PeakSignalNoiseRatio().to(device)

    x = to_tensor(image).unsqueeze(0).to(device)
    y = x + torch.randn_like(x) * sigma
    x_hat = mmse(y)

    save_image(torch.cat([y, x_hat, x], dim=-1), "nlm.png", nrow=1, padding=0)
    print(f"psnr noisy (sigma={sigma_255}): {psnr(y, x).item():.2f}")
    print(f"psnr {mmse}: {psnr(x_hat, x).item():.2f}")
