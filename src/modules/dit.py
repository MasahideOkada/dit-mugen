"""
DiT model proposed in 'Scalable Diffusion Models with Transformers'(https://arxiv.org/abs/2212.09748)
this is modified code of the original implementation (https://github.com/facebookresearch/DiT)
"""
import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as fn

from modules.utils import modulate
from modules.embeds import TimePositionEmbedding

class SegmentEmbedding(nn.Module):
    """
    embedding for (latent) inputs, corresponding to the patch embedding in ViT and DiT\n
    embeds input by 1-D convolution\n
    args:
    - `input_len`: length of input for each channel
    - `in_channels`: number of input channels
    - `embed_dim`: embedding size
    - `segment_len`: splits input into small segments whose length is this value, defalut `4`
    """
    def __init__(
        self,
        input_len: int,
        in_channels: int,
        embed_dim: int,
        segment_len: int = 4,
    ):
        assert input_len % segment_len == 0, "`input_len` must be divisible by `segment_len`"
        super().__init__()
        self.segment_len = segment_len
        self.num_segments = input_len // segment_len
        self.in_channels = in_channels
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=segment_len, stride=segment_len)

    def forward(self, x: Tensor) -> Tensor:
        """
        args:
        - `x`: 3-D tensor with shape (batches, channels, length)\n
        returns tensor with shape (batches, num segments, embed dim)
        """
        _, C, _ = x.shape
        assert C == self.in_channels,\
            f"expected {self.in_channels} as number of input channels, but found {C}"

        # B: batch size, C: num channels, L: input len, Ns: num segments, E: embed dim
        x = self.proj(x) # (B, C, L) -> (B, E, Ns)
        return x.transpose(1, 2) # (B, E, Ns) -> (B, Ns, E)

def positional_embedding(num_segments: int, embed_dim: int) -> Tensor:
    """
    positional embedding for inputs after segment embedding with shape\n
    args:
    - `num_segments`: number of segments, which is the input length after segment embedding
    - `embed_dim`: embedding size\n
    returns tensor with shape (num segments, embed dim)
    """
    emb = torch.zeros(num_segments, embed_dim)
    pos = torch.arange(0, num_segments, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) * (-np.log(10000) / embed_dim))
    emb[:, 0::2] = torch.sin(pos * div)
    emb[:, 1::2] = torch.cos(pos * div)
    emb = emb.unsqueeze(0)
    return emb

class MultiHeadAttention(nn.Module):
    """
    multi-head self attention\n
    args:
    - `embed_dim`: embedding size
    - `num_heads`: number of attention heads
    - `bias`: add bias or not, default `True`
    - `dropout`: dropout probability, default `0.0`
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        dropout: float = 0.0
    ):
        assert embed_dim % num_heads == 0, "`embed_dim` must be divisible by `num_heads`"
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.to_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.dropout = dropout
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        args:
        - `x`: 3-D tensor with shape (batches, num segments, embed dim)\n
        returns tensor with the same shapes as `x`
        """
        # B: batch size, Ns: num segments, E: embed dim, H: num heads, D: head dim(=E/H)
        B, N, E = x.shape
        qkv = (
            self.to_qkv(x)                                   # (B, Ns, E) -> (B, Ns, 3*E)
            .reshape(B, N, 3, self.num_heads, self.head_dim) # (B, Ns, 3*E) -> (B, Ns, 3, H, D)
            .permute(2, 0, 3, 1, 4)                          # (B, Ns, 3, H, D) -> (3, B, H, Ns, D)
        )
        q, k, v = qkv.unbind(0) # (3, B, H, Ns, D) -> (B, H, Ns, D) for each
        x = fn.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout) # -> (B, H, Ns, D)
        x = (
            x.transpose(1, 2) # (B, H, Ns, D) -> (B, Ns, H, D)
            .reshape(B, N, E) # (B, Ns, H, D) -> (B, Ns, E)
        )
        return self.out_proj(x) # (B, Ns, E) -> (B, Ns, E)

class DiTBlock(nn.Module):
    """
    attention block of DiT\n
    args:
    - `embed_dim`: embedding size
    - `num_heads`: number of attention heads
    - `fc_ratio`: multiplies embed size by this value for hidden size of feed forward layer, default `4.0`
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        fc_ratio: float = 4.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.attn  = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6 * embed_dim)
        )
        fc_hidden_size = int(embed_dim * fc_ratio)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, fc_hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(fc_hidden_size, embed_dim)
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        args:
        - `x`: embedded input 3-D tensor with shape (batches, num segments, embed dim)
        - `t`: embedded time 2-D tensor with shape (batches, embed dim)\n
        returns tensor with the same shape as `x`
        """
        # B: batch size, Ns: num segments, E: embed dim
        shift_mha, scale_mha, gate_mha, shift_fc, scale_fc, gate_fc = (
            self.adaLN_modulation(t) # (B, E) -> (B, 6*E)
            .chunk(6, dim=1)         # (B, 6*E) -> (B, E) for each
        )
        x += gate_mha.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_mha, scale_mha)) #  -> (B, Ns, E)
        x += gate_fc.unsqueeze(1) * self.fc(modulate(self.norm2(x), shift_fc, scale_fc)) # -> (B, Ns, E)
        return x
    
class FinalLayer(nn.Module):
    """
    final layer of DiT\n
    args:
    - `embed_dim`: embedding size
    - `segment_len`: segment length
    - `out_channels`: number of channels of outputs
    """
    def __init__(self, embed_dim: int, segment_len: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-06)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim)
        )
        self.fc = nn.Linear(embed_dim, segment_len * out_channels)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        args:
        - `x`: embedded input tensor with shape (batches, num segments, embed dim)
        - `t`: embedded time tensor with shape (batches, embed dim)\n
        returns tensor with shape (batches, num segments, output len)
        """
        # B: batch size, Ns: num segments, E: embed dim, O: output len
        shift, scale = (
            self.adaLN_modulation(t) # (B, E) -> (B, 2*E)
            .chunk(2, dim=1)         # (B, 2*E) -> (B, E) for each
        )
        x = modulate(self.norm(x), shift, scale) # (B, Ns, E) -> (B, Ns, E)
        x = self.fc(x) # (B, Ns, E) -> (B, Ns, O)
        return x

class DiT(nn.Module):
    """
    DiT architecture\n
    args:
    - `input_len`: input length of each channel
    - `in_channels`: number of input channels
    - `segment_len`: splits input into small segments whose length is this value, defalut `4`
    - `embed_dim`: embedding size, default `512`
    - `depth`: number of attention layers, default `8`
    - `num_heads`: number of attention heads, default `4`
    - `fc_ratio`: multiplies embed size by this value for hidden size of feed forward layer, default `4.0`
    """
    def __init__(
        self,
        input_len: int,
        in_channels: int,
        segment_len: int = 4,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 4,
        fc_ratio: float = 4.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.segment_len = segment_len
        self.num_heads = num_heads
        self.x_embed = SegmentEmbedding(input_len, in_channels, embed_dim, segment_len)
        self.t_embed = TimePositionEmbedding(embed_dim)
        num_segments = self.x_embed.num_segments
        self.pos_embed = nn.Parameter(torch.zeros(1, num_segments, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, fc_ratio) for _ in range(depth)
        ])
        self.final = FinalLayer(embed_dim, segment_len, self.out_channels)
        self.initialize_params()

    def initialize_params(self):
        """
        initializes model parameters
        """
        def _basic_init(module: nn.Module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = positional_embedding(*self.pos_embed.shape[1:])
        self.pos_embed.data.copy_(pos_embed.float())

        w = self.x_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embed.proj.bias, 0)

        nn.init.normal_(self.t_embed.fc[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.fc[-1].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final.fc.weight, 0)
        nn.init.constant_(self.final.fc.bias, 0)

    def unsegmentize(self, x: Tensor) -> Tensor:
        """
        turns embedded input back to original shape
        args:
        - `x`: 3-D tensor with shape (batches, num segments, embed dim)\n
        returns tensor with shape (batches, channels, length)
        """
        # B: batch size, C: num channels, L: input len, S: segment len, Ns: num segments, E: embed dim
        B, N = x.shape[0], x.shape[1]
        C = self.out_channels
        S = self.x_embed.segment_len
        x = (
            x.reshape((B, N, S, C)) # (B, Ns, E) -> (B, Ns, S, C)
            .permute(0, 3, 1, 2)    # (B, Ns, S, C) -> (B, C, Ns, S)
        )
        outputs = x.reshape((B, C, N * S)) # (B, C, Ns, S) -> (B, C, L)
        return outputs
    
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        args:
        - `x`: 3-D input tensor with shape (batches, channels, input len)
        - `t`: 1-D time tensor with shape (batches)\n
        returns tensor with the same shape as `x`
        """
        # B: batch size, C: num channels, L: input len, Ns: num segments, E: embed dim, O: output len
        x = self.x_embed(x) + self.pos_embed # (B, C, L) -> (B, Ns, E)
        t = self.t_embed(t) # (B) -> (B, E)
        for block in self.blocks:
            x = block(x, t) # (B, Ns, E) -> (B, Ns, E)
        x = self.final(x, t) # (B, Ns, E) -> (B, Ns, O)
        x = self.unsegmentize(x) # (B, Ns, O) -> (B, C, L)
        return x
