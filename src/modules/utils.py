"""
helper functions
"""
#import torch
from torch import Tensor

def modulate(x: Tensor, shift: Tensor, scale: Tensor, latent: Tensor | None = None) -> Tensor:
    """
    modulates a tensor for adaptive norm\n
    args:
    - `x`: normed input tensor with shape (bathes, num channels, embed dim)
    - `shift`: tensor with shape (batches, embed dim)
    - `scale`: tensor with shape (batches, embed dim)
    - `latent`: autoencoded latent condition with shape (batches, embed dim), optional\n
    returns tensor with the same shape as `x`
    """
    z = 1 + latent.unsqueeze(1) if isinstance(latent, Tensor) else 1
    h = (1 + scale.unsqueeze(1)) * x + shift.unsqueeze(1)
    return z * h
