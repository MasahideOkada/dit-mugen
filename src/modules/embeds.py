"""
embedding layers
"""
import numpy as np

import torch
from torch  import Tensor
import torch.nn as nn

class TimePositionEmbedding(nn.Module):
    """
    positional embedding for time\n
    args:
    - `embed_dim`: output embedding dimension
    - `freq_embed_dim`: dimension of sinusoidal position embedding, default `128`
    """
    def __init__(self, embed_dim: int, freq_embed_dim: int = 128):
        assert freq_embed_dim % 2 == 0, "`freq_embed_dim` must be divisible by 2"
        super().__init__()
        self.embed_dim = embed_dim
        self.freq_embed_dim = freq_embed_dim
        self.fc = nn.Sequential(
            nn.Linear(freq_embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, t: Tensor) -> Tensor:
        """
        args:
        - `t`: 1-D tensor with shape (batches)\n
        returns tensor with shape (batchs, embed dim)
        """
        # B: batch size, F: freq dim, E: embed dim 
        half_freq_dim = self.freq_embed_dim // 2
        freq = np.log(10000) / (half_freq_dim - 1) 
        freq = torch.exp(torch.arange(half_freq_dim, dtype=torch.float, device=t.device) * -freq) # -> (F/2)
        freq = t[:, None] * freq[None, :] # (F/2) -> (B, F/2)
        freq = torch.cat((freq.sin(), freq.cos()), dim=-1) # (B, F/2) -> (B, F)
        emb = self.fc(freq) # (B, F) -> (B, E)
        return emb
