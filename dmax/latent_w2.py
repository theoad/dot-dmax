from typing import Optional, Literal, Tuple, Union, List
from types import MethodType
from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, DoubleTensor, BoolTensor
from torch.distributions import MultivariateNormal
from torchvision.transforms import Normalize
from torchmetrics import Metric
from diffusers.models import AutoencoderKL


STABILITY_CONST = 1e-8


def unsqueeze_like(tensor: Tensor, like: Tensor) -> Tensor:
    """
    Unsqueeze last dimensions of tensor to match another tensor's number of dimensions.

    Args:
        tensor (torch.Tensor): tensor to unsqueeze
        like (torch.Tensor): tensor whose dimensions to match
    """
    n_unsqueezes = like.ndim - tensor.ndim
    if n_unsqueezes < 0:
        raise ValueError(f"tensor.ndim={tensor.ndim} > like.ndim={like.ndim}")
    elif n_unsqueezes == 0:
        return tensor
    else:
        return tensor[(...,) + (None,) * n_unsqueezes]


# Function from the pyRiemann package ported in pytorch
def matrix_operator(matrices: Tensor, operator) -> Tensor:
    """
    Matrix equivalent of an operator. Works batch-wise
    Porting of pyRiemann to pyTorch
    Original Author: Alexandre Barachant
    https://github.com/alexandrebarachant/pyRiemann
    """
    eigvals, eigvects = torch.linalg.eigh(matrices, UPLO='L')
    eigvals = torch.diag_embed(operator(eigvals))
    return eigvects @ eigvals @ eigvects.transpose(-2, -1)


def eye_like(matrices: Tensor) -> Tensor:
    """
    Return Identity matrix with the same shape, device and dtype as matrices

    :param matrices: Batch of matrices with shape [*, C, D] where * is zero or leading batch dimensions
    :return: Tensor T with shape [*, C, D]. with T[i] = torch.eye(C, D)
    """
    return torch.eye(*matrices.shape[-2:-1], out=torch.empty_like(matrices)).expand_as(matrices)


def sqrtm(matrices: Tensor) -> Tensor:
    """
    :param matrices: batch of SPSD matrices
    :returns: batch containing mat. square root of each matrix
    """
    return matrix_operator(matrices, torch.sqrt)


def invsqrtm(matrices: Tensor) -> Tensor:
    """
    :param matrices: batch of SPD matrices
    :returns: batch containing mat. inverse sqrt. of each matrix
    """
    isqrt = lambda x: 1. / torch.sqrt(x)
    return matrix_operator(matrices, isqrt)


def is_symmetric(matrices: Tensor) -> BoolTensor:
    """
    Boolean method. Checks if matrix is symmetric.

    :param matrices: Batch of matrices with shape [*, D, D] where * is zero or leading batch dimensions
    :return: Boolean tensor T with shape [*]. with T[i] == True <=> matrices[i] is symmetric
    """
    if matrices.size(-1) != matrices.size(-2):
        return torch.full_like(matrices.mean(dim=(-1, -2)), 0).bool()  # = Tensor([False, False, ..., False])
    return torch.sum((matrices - matrices.transpose(-2, -1))**2, dim=(-1, -2)) < STABILITY_CONST  # noqa


def min_eig(matrices: Tensor) -> Tensor:
    """
    Returns the minimal eigen values of a batch of matrices (signed).

    :param matrices: Batch of matrices with shape [*, D, D] where * is zero or leading batch dimensions
    :return: Tensor T with shape [*]. with T[i] = min(eig(matrices[i]))
    """
    return torch.linalg.eigvalsh(matrices)[..., 0]


def is_pd(matrices: Tensor, strict=True, eps=STABILITY_CONST) -> BoolTensor:
    """
    Boolean method. Checks if matrices are Positive Definite (PD).

    :param matrices: Batch of matrices with shape [*, D, D] where * is zero or leading batch dimensions
    :param strict: If ``False`` checks the matrices are positive semi-definite
    :param eps: tolerance above which eigenvalue is considered strictly positive
    :return: Boolean tensor T with shape [*]. with T[i] == True <=> matrices[i] is PD
    """
    return min_eig(matrices) > eps if strict else min_eig(matrices) >= 0


def is_spd(matrices: Tensor, strict=True, eps=STABILITY_CONST) -> BoolTensor:
    """
    Boolean method. Checks if matrices are Symmetric and Positive Definite (SPD).

    :param matrices: Batch of matrices with shape [*, D, D] where * is zero or leading batch dimensions
    :param strict: If ``False`` checks the matrices are positive semi-definite (SPSD)
    :param eps: tolerance above which eigenvalue is considered strictly positive
    :return: Boolean tensor T with shape [*]. with T[i] == True <=> matrices[i] is SPD
    """
    return torch.logical_and(is_symmetric(matrices), is_pd(matrices, strict=strict, eps=eps)).bool()


def make_psd(
        matrices: Tensor,
        strict: bool = False,
        return_correction: bool = False,
        diag: bool = False,
        eps: float = STABILITY_CONST
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Add to each matrix its minimal eigen value to make it positive definite.

    :param matrices: Batch of matrices with shape [*, D, D] where * is zero or leading batch dimensions
    :param strict: If ``True``, add a small stability constant to make the matrices positive definite (PD)
    :param return_correction: If ``True``, returns the correction added to the diagonal of the matrices.
    :param diag: If ``True``, considers input as diagonal matrices
    :param eps: tolerance above which eigenvalue is considered strictly positive
    :return: Tensor T with shape [*]. with T[i] = matrices[i] + min(eig(matrices[i]) * I
    """
    smallest_eig = matrices.min(-1)[0] if diag else min_eig(matrices)
    small_positive_val = smallest_eig.clamp(max=0).abs()
    if strict: small_positive_val += eps
    if diag:
        res = matrices + small_positive_val[..., None]
    else:
        I = eye_like(matrices)
        res = matrices + I * small_positive_val[..., None, None]
    if return_correction:
        return res, small_positive_val
    return res


def mean_cov(sum: Tensor, sum_corr: Tensor, num_obs: Union[Tensor, int], diag: bool = False) -> Tuple[Tensor, Tensor]:
    """
    Empirical computation of mean and covariance matrix

    :param sum: Sum of feature vectors of shape [*, D]
    :param sum_corr: Sum of covariance matrices of shape [*, D, D] ([*, D] if `diag`==True)
    :param num_obs: Number of observations
    :param diag: If ``True``, will expect the covariance to be a vector of variance
    :return: The features mean and covariance of shape [*, D] and [*, D, D] ([*, D] if `diag`==True)
    """
    mean = sum / unsqueeze_like(num_obs, sum)
    cov = sum_corr / unsqueeze_like(num_obs, sum_corr)
    cov -= mean ** 2 if diag else mean.unsqueeze(-1) @ mean.unsqueeze(-2)
    return mean, cov


def w2_gaussian(
        mean_source: Tensor,
        mean_target: Tensor,
        cov_source: Tensor,
        cov_target: Tensor,
        eps: float = 1e-5
) -> Tensor:
    """
    Computes closed form squared W2 distance between Gaussian distribution_models (also known as Gelbrich Distance)
    :param mean_source: A 1-dim vectors representing the source distribution mean with optional leading batch dims [*, D]
    :param mean_target: A 1-dim vectors representing the target distribution mean with optional leading batch dims [*, D]
    :param cov_source: A 2-dim matrix representing the source distribution covariance [*, D, D]
    :param cov_target: A 2-dim matrix representing the target distribution covariance [*, D, D]
    :param eps: small positive stability constant

    :return: The squared Wasserstein 2 distance between N(mean_source, cov_source) and N(mean_target, cov_target)
    """
    cov_source = make_psd(cov_source, strict=True, eps=eps)
    cov_target = make_psd(cov_target, strict=True, eps=eps)
    cov_target_sqrt = sqrtm(cov_target)
    mix = make_psd(cov_target_sqrt @ cov_source @ cov_target_sqrt, strict=False, eps=eps)

    mean_shift = torch.sum((mean_source - mean_target) ** 2, dim=-1)
    cov_shift_trace = torch.diagonal(cov_source + cov_target - 2 * sqrtm(mix), dim1=-2, dim2=-1).sum(dim=-1)
    return mean_shift + cov_shift_trace


def gaussian_transport_operators(cov_source: Tensor, cov_target: Tensor, eps: float = 1e-5) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Batch implementation of eq. 17, 19 in [1]

    .. math::

        (17) T_{s \longrightarrow t} = (1 - P/G^{*}) \Sigma^{-0.5}_{s} (\Sigma^{0.5}_{s} \Sigma_{t}
         \Sigma^{0.5}_{s})^{0.5} \Sigma^{-0.5}_{s} + P/G^{*} I

        (19) T_{s \longrightarrow t} = (1 - P/G^{*}) \Sigma^{+0.5}_{t} (\Sigma^{0.5}_{t} \Sigma_{s}
         \Sigma^{0.5}_{t})^{0.5} \Sigma^{-0.5}_{t} \Sigma^{\dagger}_{s} + P/G^{*} I

             \Sigma_w = \Sigma^{0.5}_{t} (I - \Sigma^{0.5}_{t} T^{*} \Sigma^{\dagger}_{s} T^{*}
              \Sigma^{0.5}_{t}) \Sigma^{0.5}_{t},  T^{*} = T_{t \longrightarrow s} (17)

    [1] D. Freirich, T. Michaeli and R. Meir.
    `A Theory of the Distortion-Perception Tradeoff in Wasserstein Space <https://proceedings.neurips.cc/paper/2021/
    hash/d77e68596c15c53c2a33ad143739902d-Abstract.html>`_

    :param cov_source: Batch of SPD matrices. Source covariances. [*, D1, D1]
    :param cov_target: Batch of SPSD matrices. Target covariances. [*, D2, D2]
    :param eps: small positive stability constant
    :return: Batch of transport operators T_{s -> t} and \Sigma_w
    """
    I = eye_like(cov_source)

    if is_spd(cov_source, strict=True, eps=eps).all():
        sqrtCs, isqrtCs = sqrtm(cov_source), invsqrtm(cov_source)
        T0 = isqrtCs @ sqrtm(make_psd(sqrtCs @ cov_target @ sqrtCs, strict=True, eps=eps)) @ isqrtCs
        Cw0 = None
    else:
        sqrtCt, isqrtCt = sqrtm(cov_target), invsqrtm(make_psd(cov_target, strict=True, eps=eps))
        mix = sqrtm(make_psd(sqrtCt @ cov_source @ sqrtCt, strict=False, eps=eps))
        T_star = isqrtCt @ mix @ isqrtCt
        pinvCs = torch.linalg.pinv(cov_source, atol=eps ** 0.5)
        T0 = (sqrtCt @ mix @ isqrtCt @ pinvCs)
        Cw0 = sqrtCt @ (I - sqrtCt @ T_star @ pinvCs @ T_star @ sqrtCt) @ sqrtCt

    assert T0.shape == cov_source.shape
    return T0, Cw0


def transport(features: Tensor, mean_source: Tensor, mean_target: Tensor, T0: Tensor, Cw0: Optional[Tensor], pg_star: float = 0.) -> Tensor:
    r"""
    Executes optimal W2 transport of the sample given as `features` using the provided mean and transport operators (T, Cw)
    in batch fashion according to eq. 17 and eq. 19 in [1]

    .. math::

        (17) T_{s \longrightarrow t} = (1 - P/G^{*}) \Sigma^{-0.5}_{s} (\Sigma^{0.5}_{s} \Sigma_{t}
         \Sigma^{0.5}_{s})^{0.5} \Sigma^{-0.5}_{s} + P/G^{*} I

        (19) T_{s \longrightarrow t} = (1 - P/G^{*}) \Sigma^{+0.5}_{t} (\Sigma^{0.5}_{t} \Sigma_{s}
         \Sigma^{0.5}_{t})^{0.5} \Sigma^{-0.5}_{t} \Sigma^{\dagger}_{s} + P/G^{*} I

             \Sigma_w = \Sigma^{0.5}_{t} (I - \Sigma^{0.5}_{t} T^{*} \Sigma^{\dagger}_{s} T^{*}
              \Sigma^{0.5}_{t}) \Sigma^{0.5}_{t},  T^{*} = T_{t \longrightarrow s} (17)

    [1] D. Freirich, T. Michaeli and R. Meir.
    `A Theory of the Distortion-Perception Tradeoff in Wasserstein Space <https://proceedings.neurips.cc/paper_files/paper/2021/hash/d77e68596c15c53c2a33ad143739902d-Abstract.html>`_

    :param features: Batch of samples from the source distribution, to transport to the target distribution. [*, B, D1]
    :param mean_source: Mean of the source distribution. [*, D1]
    :param mean_target: Mean of the target distribution. [*, D2]
    :param T0: transport Operator from the source to the target distribution. [*, D2, D1]
    :param Cw0: Noise covariance if the source distribution is degenerate. [*, D2, D1]

    :return: T (input - mean_source) + mean_target + W,   W~Normal(0, Cw). [*, D2]
    """
    T = (1 - pg_star) * T0 + pg_star * eye_like(T0)
    features_centered = (features - mean_source[..., None, :])
    transported = (T @ features_centered[..., None]).squeeze() + mean_target[..., None, :]
    if Cw0 is not None and pg_star != 1:
        noise = MultivariateNormal(torch.zeros_like(mean_target), Cw0 * (1 - pg_star) ** 0.5)
        transported += noise.sample()[..., None, :]
    return transported


def pair(t) -> Tuple:
    if isinstance(t, tuple): return t
    return t, t


class LatentW2(Metric):
    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False

    source_features: Union[DoubleTensor, List[DoubleTensor]]
    source_features_sum: DoubleTensor
    source_features_cov_sum: DoubleTensor
    source_features_num_samples: DoubleTensor

    target_features: Union[DoubleTensor, List[DoubleTensor]]
    target_features_sum: DoubleTensor
    target_features_cov_sum: DoubleTensor
    target_features_num_samples: DoubleTensor

    mean_source: Optional[DoubleTensor]
    mean_target: Optional[DoubleTensor]
    cov_source: Optional[DoubleTensor]
    cov_target: Optional[DoubleTensor]
    transport_operator: Optional[DoubleTensor]
    noise_covariance: Optional[DoubleTensor]

    def __init__(
            self,
            vae: Optional[str] = "stabilityai/sd-vae-ft-mse",
            block_size: Union[int, Tuple[int, int]] = 3,  # h,w format
            folding_kernel_size: Union[int, Tuple[int, int]] = 3,  # h,w format
            folding_kernel_sigma: Union[float, Tuple[float, float]] = float('inf'),  # h,w format
            shared: bool = True,
            resolution: Optional[Union[int, Tuple[int, int]]] = None,  # ignored if shared == False
            eps: float = 1e-5,
            reset_target_features: bool = False,
            store_features: bool = False
    ) -> None:
        """
        Transports a `source [image] distribution` to a `target [image] distribution` in the latent space of the provided VAE.
        We use a closed-form solution to optimal transport between mutli-variate Gaussians.

        >>> dmax = LatentW2("stabilityai/sd-vae-ft-ema")
        >>> dmax.update(torch.randn(10, 3, 512, 480), "source")  # supports different aspect-ratio
        >>> dmax.update(torch.randn(10, 3, 384, 504), "target")  # supports different resolutions
        >>> dmax.compute()  # computes the tranpsort operator (+ synchronises in distributed environement)
        >>> transported = dmax(torch.randn(10, 3, 256, 376))

        :param vae: Version of VAE (currently hugging-face provides "stabilityai/sd-vae-ft-mse" and "stabilityai/sd-vae-ft-ema")
        :param block_size: The patch-size `p` in latent space. All overlapping patches of size (c,p,p) will be transported.
        :param folding_kernel_size: The wieghting kernel used when averaging back the transported overlapping patches.
        :param folding_kernel_sigma: The standard deviation of the weighting kernel `folding_kernel_size`.
        :param shared: Whether or not the have a different transport operator for each patch.
        :param resolution: The image resolution (must be provided when `shared` == False).
        :param eps: Small stability value which helps computation when covariance matrices are close to singular.
        :param reset_target_features: If `False` will not reset the target image features when `reset()` method is called.
        :param store_features: If `True` will store all the updated features in a buffer (large memory footprint).
        """
        super().__init__()
        if not shared and resolution is None:
            raise ValueError("`resolution` must be set if patch distribution is not `shared`")
        if any(f > b for f, b in zip(pair(folding_kernel_size), pair(block_size))):
            raise ValueError("`folding_kernel` must be smaller than the `block_size`")
        if any(not f % 2 or not b % 2 for f, b in zip(pair(folding_kernel_size), pair(block_size))):
            raise ValueError("`block_size` and `folding_kernel` must not divide by 2")
        if not shared:
            raise ValueError("`shared`==False is not yet supported")

        self.block_size = pair(block_size)
        self.folding_kernel_size = pair(folding_kernel_size)
        self.folding_kernel_sigma = pair(folding_kernel_sigma)
        self.reset_target_features = reset_target_features
        self.store_features = store_features
        self.shared = shared
        self.eps = eps

        if vae:
            self.vae = AutoencoderKL.from_pretrained(vae)
            self.vae.train = MethodType(lambda *args: None, self.vae)  # makes sure vae doesn't exit eval mode
            self.vae_scale_factor = int(2 ** (len(self.vae.block_out_channels) - 1))
            self.vae_latent_dim = self.vae.post_quant_conv.weight.size(1)
        else:
            self.vae = None
            self.vae_scale_factor = 1
            self.vae_latent_dim = 3

        if self.store_features:
            self.add_state("source_features", [], dist_reduce_fx=None, persistent=True)
            self.add_state("target_features", [], dist_reduce_fx=None, persistent=True)
        else:
            nparams = int(np.prod(resolution)) // (self.vae_scale_factor ** 2) if resolution else None
            num_features = self.vae_latent_dim * int(np.prod(self.block_size))
            n_nb_feets = 0 if shared else nparams
            vec_nb_feets = (num_features,) if shared else (nparams, num_features)
            mx_nb_feets = (num_features, num_features) if shared else (nparams, num_features, num_features)
            self.add_state("source_features_sum", torch.zeros(vec_nb_feets).double(), dist_reduce_fx="sum", persistent=True)
            self.add_state("source_features_cov_sum", torch.zeros(mx_nb_feets).double(), dist_reduce_fx="sum", persistent=True)
            self.add_state("source_features_num_samples", torch.tensor(n_nb_feets).long(), dist_reduce_fx="sum", persistent=True)

            self.add_state("target_features_sum", torch.zeros(vec_nb_feets).double(), dist_reduce_fx="sum", persistent=True)
            self.add_state("target_features_cov_sum", torch.zeros(mx_nb_feets).double(), dist_reduce_fx="sum", persistent=True)
            self.add_state("target_features_num_samples", torch.tensor(n_nb_feets).long(), dist_reduce_fx="sum", persistent=True)

        self.register_buffer("mean_source", None)
        self.register_buffer("mean_target", None)
        self.register_buffer("cov_source", None)
        self.register_buffer("cov_target", None)
        self.register_buffer("transport_operator", None)
        self.register_buffer("noise_covariance", None)

        mean, std = (0.5,) * 3, (0.5,) * 3
        self.normalize = Normalize(mean, std)
        self.denormalize = nn.Sequential(
            Normalize(torch.as_tensor(mean).zero_(), 1. / torch.as_tensor(std)),
            Normalize(-torch.as_tensor(mean), torch.as_tensor(std).zero_() + 1.)
        )

    @staticmethod
    def _get_gaussian_kernel1d(kernel_size: int, sigma: float) -> Tensor:
        ksize_half = (kernel_size - 1) * 0.5
        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel1d = pdf / pdf.sum()
        return kernel1d

    @staticmethod
    def _get_gaussian_kernel2d(
            kernel_size: Union[int, Tuple[int, int]],
            sigma: Union[float, Tuple[float, float]],
            dtype: torch.dtype = torch.double,
            device: Optional[torch.device] = None
    ) -> Tensor:
        kernel_size = pair(kernel_size)
        sigma = pair(sigma)
        kernel1d_x = LatentW2._get_gaussian_kernel1d(kernel_size[0], sigma[0]).to(device=device, dtype=dtype)
        kernel1d_y = LatentW2._get_gaussian_kernel1d(kernel_size[1], sigma[1]).to(device=device, dtype=dtype)
        kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
        return kernel2d

    @property
    def gaussian_weights(self):
        # weighted folding according to Gaussian folding kernel
        weights = self._get_gaussian_kernel2d(self.folding_kernel_size, self.folding_kernel_sigma)  # k, k
        padding = ((self.block_size[1] - self.folding_kernel_size[1]) // 2,) * 2 + ((self.block_size[0] - self.folding_kernel_size[0]) // 2,) * 2  # left, right, top, bottom
        weights = F.pad(weights, padding, mode="constant", value=0.).flatten().repeat(self.vae_latent_dim)  # CxPxP
        return weights

    def encode(self, images: torch.FloatTensor) -> torch.DoubleTensor:
        images_normalized = self.normalize(images)
        latents = self.vae.encode(images_normalized).latent_dist.sample() if self.vae else images_normalized
        return latents.double()

    def decode(self, latents: torch.DoubleTensor) -> torch.FloatTensor:
        images_normalized = self.vae.decode(latents.float()).sample if self.vae else latents.float()
        images = self.denormalize(images_normalized)
        return images

    def patchify(self, latents: torch.DoubleTensor, pad: bool = True) -> torch.DoubleTensor:
        if pad or not self.shared:
            padding = (self.block_size[1] // 2,) * 2 + (self.block_size[0] // 2,) * 2  # left, right, top, bottom
            latents = F.pad(latents, padding, mode="replicate")
        all_overlapping_patches = F.unfold(latents, self.block_size)  # B, CxPxP, HxW
        all_overlapping_patches = all_overlapping_patches.permute(2, 0, 1)  # HxW, B, CxPxP
        if self.shared:
            all_overlapping_patches = all_overlapping_patches.flatten(0, 1)  # BxHxW, CxPxP
        return all_overlapping_patches

    def unpatchify(self, patches: torch.DoubleTensor, size: torch.Size) -> torch.DoubleTensor:
        b, c, h, w = size
        divisor = self.patchify(torch.ones((1, c, h, w), device=patches.device).double())
        if self.shared:
            patches = patches.unflatten(0, (h * w, b))  # HxW, B, CxPxP
            divisor = divisor.unflatten(0, (h * w, 1))  # HxW, 1, CxPxP

        weights = self.gaussian_weights[None, None].to(patches.device)
        patches *= weights  # HxW, B, CxPxP
        divisor *= weights  # HxW, 1, CxPxP
        patches = patches.permute(1, 2, 0)  # B, CxPxP, HxW
        divisor = divisor.permute(1, 2, 0)  # 1, CxPxP, HxW
        latents = F.fold(patches, (h, w), self.block_size, padding=(self.block_size[0] // 2, self.block_size[1] // 2))  # B, C, H, W
        divisor = F.fold(divisor, (h, w), self.block_size, padding=(self.block_size[0] // 2, self.block_size[1] // 2))  # 1, C, H, W
        return latents / divisor

    def update(self, images: torch.FloatTensor, distribution: Literal["source", "target"] = "source", encode: bool = True) -> None:
        if encode:
            latents = self.encode(images)
            samples = self.patchify(latents, pad=False)
        else:
            samples = images.double()
        if distribution == "source":
            if self.store_features:
                self.source_features.append(samples)
            else:
                self.source_features_sum += samples.sum(dim=-2)
                self.source_features_cov_sum += torch.einsum("...bi,...bj->...ij", samples, samples)
                self.source_features_num_samples += samples.size(-2)
        elif distribution == "target":
            if self.store_features:
                self.target_features.append(samples)
            else:
                self.target_features_sum += samples.sum(dim=-2)
                self.target_features_cov_sum += torch.einsum("...bi,...bj->...ij", samples, samples)
                self.target_features_num_samples += samples.size(-2)
        else:
            raise NotImplementedError()

    @staticmethod
    def _mean_cov(sum, sum_corr, n):
        mean = sum / n[..., None]
        cov = sum_corr - n[..., None, None] * mean[..., None] @ mean[..., None, :]
        cov /= n[..., None, None] - 1
        return mean, cov

    def compute(self) -> Tensor:
        if self.store_features:
            if isinstance(self.source_features, list):
                self.source_features = torch.cat(self.source_features, dim=-2).double()
            if isinstance(self.target_features, list):
                self.target_features = torch.cat(self.target_features, dim=-2).double()
            sf, tf = self.source_features, self.target_features
            self.mean_source = sf.mean(-2, keepdim=True)
            self.mean_target = tf.mean(-2, keepdim=True)

            sd, td = sf - self.mean_source, tf - self.mean_target
            self.cov_source = 1.0 / (sf.size(-2) - 1) * torch.einsum("...bi,...bj->...ij", sd, sd)
            self.cov_target = 1.0 / (tf.size(-2) - 1) * torch.einsum("...bi,...bj->...ij", td, td)
            self.mean_source = self.mean_source.squeeze(-2)
            self.mean_target = self.mean_target.squeeze(-2)

        else:
            self.mean_source, self.cov_source = self._mean_cov(
                self.source_features_sum,
                self.source_features_cov_sum,
                self.source_features_num_samples
            )
            self.mean_target, self.cov_target = self._mean_cov(
                self.target_features_sum,
                self.target_features_cov_sum,
                self.target_features_num_samples
            )

        self.transport_operator, self.noise_covariance = gaussian_transport_operators(self.cov_source, self.cov_target, eps=self.eps)
        return w2_gaussian(self.mean_source, self.mean_target, self.cov_source, self.cov_target, eps=self.eps)

    def reset(self) -> None:
        self.transport_operator = None
        self.noise_covariance = None
        self.mean_source = None
        self.mean_target = None
        self.cov_source = None
        self.cov_target = None

        if not self.reset_target_features:
            target_features_sum = deepcopy(self.target_features_sum)
            target_features_cov_sum = deepcopy(self.target_features_cov_sum)
            target_features_num_samples = deepcopy(self.target_features_num_samples)
            super().reset()
            self.target_features_sum = target_features_sum
            self.target_features_cov_sum = target_features_cov_sum
            self.target_features_num_samples = target_features_num_samples
        else:
            super().reset()

    def transport(self, images: torch.FloatTensor, zero_shot: bool = False, pg_star: float = 0.) -> torch.FloatTensor:
        latents = self.encode(images)
        patches = self.patchify(latents)

        if zero_shot:
            assert not self.reset_target_features, "zero shot makes no sense when target features are reset."
            self.reset()
            self.update(patches, distribution="source", encode=False)  # noqa

        if self.transport_operator is None: self.compute()

        patches_transported = transport(
            patches, self.mean_source, self.mean_target, self.transport_operator, self.noise_covariance, pg_star=pg_star
        )

        latents_transported = self.unpatchify(patches_transported, latents.shape)  # noqa
        transported = self.decode(latents_transported)
        return transported

    def forward(self, images: torch.FloatTensor, zero_shot: bool = False, pg_star: float = 0.) -> torch.FloatTensor:
        return self.transport(images)

    def extra_repr(self) -> str:
        return f"block_size={self.block_size}, shared={self.shared}, eps={self.eps}"
