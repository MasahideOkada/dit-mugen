"""
ddpm model with ddim sampler
"""
from typing import Any, Callable

import numpy as np

import torch
from torch import Tensor
import torch.nn as nn

def linear_noise_scheduler(
    start: float = 0.0001,
    end: float = 0.02
) -> Callable[[Tensor], tuple[Tensor, Tensor, Tensor]]:
    """
    linear noise schedule\n
    args:
    - `start`: starting noise, default `0.0001`
    - `end`: ending noise, default `0.02`\n
    returns noise scheduling function
    """
    assert 0 <= start and start <= end and end <= 1.0,\
        "`start` and `end` must satisfy 0 <= `start` <= `end` <= 1"

    def scheduler(t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        noise scheduling function\n
        args:
        - `t`: time tensor with shape (batches)\n
        returns 3 tensors for diffusion coefficients
        """
        sigmas_squared = (end - start) * t + start
        sigmas = torch.sqrt(sigmas_squared)
        alphas = torch.sqrt(1.0 - sigmas_squared)
        # snr = alpha^2 / sigma^2 = (1 - sigma^2) / sigma^2 = (1 / sigma^2) - 1
        snrs = (1.0 / torch.clamp(sigmas_squared, min=1e-20)) - 1.0
        return alphas, sigmas, snrs

    return scheduler

def cosine_noise_scheduler(
    name: str | None,
    a: float | None = None,
    b: float | None = None
) -> Callable[[Tensor], tuple[Tensor, Tensor, Tensor]]:
    """
    cosine noise schedule where alpha = cos(a * t + b)\n
    2 names of schedule are available:\n
    one used in 'Moûsai: Text-to-Music Generation with Long-Context Latent Diffusion'(https://arxiv.org/abs/2301.11757)\n
    the other used in 'Noise2Music: Text-conditioned Music Generation with Diffusion Models'(https://arxiv.org/abs/2302.03917)\n
    args:
    - `name`: name of schedule, 'mousai', 'noise2music', or `None`.
              if `None`, `a` and `b` must be specified
    - `a`: constant value, a number or `None`
    - `b`: constant value, a number or `None`\n
    returns noise scheduling function
    """
    if name is None:
        assert isinstance(a, (int, float)) and isinstance(b, (int, float)),\
            "values for `a` and `b` must be specified when `name` is None"
        a = float(a)
        b = float(b)
    elif name == "mousai":
        a = 0.5 * np.pi
        b = 0.0
    elif name == "noise2music":
        a = np.arctan(np.e ** 10) - np.arctan(np.e ** (-10))
        b = np.arctan(np.e ** (-10))
    else:
        raise Exception(f"unknown scheduler `{name}`")
    
    def scheduler(t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        noise scheduling function\n
        args:
        - `t`: time tensor with shape (batches)\n
        returns 3 tensors for diffusion coefficients
        """
        alphas = torch.cos(a * t + b)
        sigmas = torch.sin(a * t + b)
        # snr = alpha^2 / sigma^2 = (1 - sigma^2) / sigma^2 = (1 / sigma^2) - 1
        snrs = (1.0 / torch.clamp(sigmas ** 2, min=1e-20)) - 1.0
        return alphas, sigmas, snrs

    return scheduler

def get_scheduler(name: str | None, **kwargs) -> Callable[[Tensor], tuple[Tensor, Tensor, Tensor]]:
    """
    scheduler function specified by `name`\n
    arguments for `linear_noise_scheduler` or `cosine_noise_scheduler` can be passed as kwargs 
    if you want to specify them\n
    args:
    - `name`: name of schedule, 'linear', 'mousai', 'noise2music', or `None`
              if `None`, it is considered as cosine schedule\n
    returns noise scheduling function
    """
    name = name.lower() if isinstance(name, str) else name
    if name == "linear":
        return linear_noise_scheduler(**kwargs)
    else:
        return cosine_noise_scheduler(name, **kwargs)

class GaussianDiffusion(nn.Module):
    """
    velocity-objective diffusion model proposed in \n
    'Progressive Distillation for Fast Sampling of Diffusion Models'(https://arxiv.org/abs/2202.00512)\n
    with Min-SNR-gamma loss weighting proposed in \n
    'Efficient Diffusion Training via Min-SNR Weighting Strategy'(https://arxiv.org/abs/2303.09556)\n
    args:
    - `net`: neural network model to predict velocity
    - `loss_fn`: loss function with `reduction`='none'
    - `scheduler_name`: noise scheduler name, 'linear', 'mousai', or 'noise2music'
    - `snr_gamma`: constant value in Min-SNR-gamma strategy to calculate loss weights, default `5.0`
    """
    def __init__(
        self,
        net: nn.Module,
        loss_fn: Any,
        scheduler_name: str,
        snr_gamma: float = 5.0
    ):
        super().__init__()
        self.net = net
        self.loss_fn = loss_fn
        self.scheduler = get_scheduler(scheduler_name)
        self.snr_gamma = snr_gamma
    
    def forward(self, x: Tensor) -> Tensor:
        """
        calculates loss by predicting velocity\n
        args:
        - `x`: input 3-D tensor with shape (batches, num channels, length)\n
        returns the mean of weighted losses
        """
        # B: batch size, C: num channels, L: input len
        B, device = x.shape[0], x.device
        t = torch.rand(B, device=device) # -> (B)
        alphas, sigmas, snrs = self.scheduler(t) # (B) -> (B) for each
        alphas = alphas.view(B, *((1,) * (len(x.shape) - 1))) # (B) -> (B, 1, 1)
        sigmas = sigmas.view(B, *((1,) * (len(x.shape) - 1))) # (B) -> (B, 1, 1)
        # Min-SNR-gamma weighting
        loss_weights = torch.clamp(snrs, max=self.snr_gamma) / (snrs + 1) # -> (B)

        eps = torch.randn_like(x) # -> (B, C, L)
        x_t = alphas * x + sigmas * eps # -> (B, C, L)
        # target velocity
        targets = alphas * eps - sigmas * x # -> (B, C, L)
        # predicted velocity
        preds = self.net(x_t, t) # -> (B, C, L)

        loss = self.loss_fn(preds, targets) # -> (B, C, L)
        loss = loss_weights * loss.mean(dim=[1, 2]) # (B, C, L) -> (B)
        return loss.mean()
    
    @torch.no_grad()
    def sample(self, shape: tuple[int, int], steps: int, device: Any) -> Tensor:
        """
        ddim sampler\n
        args:
        - `shape`: shape of data to sample, without batch dimension
        - `steps`: number of denoising steps
        - `device`: torch.device\n
        returns sampled tensor with shape specified by `shape`
        """
        # C: num channels, L: input len, T: time steps

        # make denoising time table for T and T-1
        time_table = np.linspace(1.0, 0.0, steps + 1)
        # T
        current_T = time_table[:-1]
        # T-1
        next_T = time_table[1:]

        scheduler = linear_noise_scheduler(0.0, 1.0)

        # denoising process
        x = torch.randn(shape, device=device).unsqueeze(0) # -> (1, C, L)
        for T_c, T_n in zip(current_T, next_T):
            t_c = torch.tensor([T_c], device=device) # -> (1)
            alpha_c, sigma_c, _ = scheduler(t_c) # -> (1) for each
            alpha_c = alpha_c.view(1, *((1,) * (len(x.shape) - 1))) # (1) -> (1, 1, 1)
            sigma_c = sigma_c.view(1, *((1,) * (len(x.shape) - 1))) # (1) -> (1, 1, 1)

            v = self.net(x, t_c) # -> (1, C, L)
            x_0 = alpha_c * x - sigma_c * v # -> (1, C, L)
            eps = sigma_c * x + alpha_c * v # -> (1, C, L)

            t_n = torch.tensor([T_n], device=device) # -> (1)
            alpha_n, sigma_n, _ = scheduler(t_n) # -> (1) for each
            alpha_n = alpha_n.view(1, *((1,) * (len(x.shape) - 1))) # (1) -> (1, 1, 1)
            sigma_n = sigma_n.view(1, *((1,) * (len(x.shape) - 1))) # (1) -> (1, 1, 1)

            # update x
            x = alpha_n * x_0 + sigma_n * eps
        return x.squeeze(0) # (1, C, L) -> (C, L)
