# Reference:
#   Z. Bang et al., "USEF-TSE: Universal Speaker Embedding Free Target Speaker Extraction"
# Based on the implementation from:
#   https://github.com/ZBang/USEF-TSE

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

from wesep.modules.common.get_layer_from_string import get_layer


class USEF_attentionblock(nn.Module):
    """
        https://github.com/ZBang/USEF-TSE/blob/main/models/local/TFgridnet.py
    """

    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        emb_dim,  # 128
        n_freqs,  # 65
        n_head,  # 4
        approx_qk_dim,
        activation="prelu",
        eps=1e-5,
    ):
        super().__init__()
        assert activation == "prelu"

        E = math.ceil(
            approx_qk_dim * 1.0 /
            n_freqs)  # approx_qk_dim is only approximate  512 / 65 = 8
        assert emb_dim % n_head == 0

        self.add_module("attn_conv_Q", nn.Conv2d(emb_dim, n_head * E, 1))
        self.add_module(
            "attn_norm_Q",
            AllHeadPReLULayerNormalization4DCF((n_head, E, n_freqs), eps=eps),
        )

        self.add_module("attn_conv_K", nn.Conv2d(emb_dim, n_head * E, 1))
        self.add_module(
            "attn_norm_K",
            AllHeadPReLULayerNormalization4DCF((n_head, E, n_freqs), eps=eps),
        )

        self.add_module("attn_conv_V",
                        nn.Conv2d(emb_dim, n_head * emb_dim // n_head, 1))
        self.add_module(
            "attn_norm_V",
            AllHeadPReLULayerNormalization4DCF(
                (n_head, emb_dim // n_head, n_freqs), eps=eps),
        )

        self.add_module(
            "attn_concat_proj",
            nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim, 1),
                get_layer(activation)(),
                LayerNormalization4DCF((emb_dim, n_freqs), eps=eps),
            ),
        )

        self.n_head = n_head

    def forward(self, batch, aux):
        """GridNetV2Block Forward.

        Args:
            x: [B, C, T, Q]
            out: [B, C, T, Q]
        """

        B, _, old_T, old_Q = batch.shape
        aux_T = aux.shape[-2]

        Q = self["attn_norm_Q"](self["attn_conv_Q"](
            batch))  # [B, n_head, C, T, Q], [B, 4, 8, T, 65]
        K = self["attn_norm_K"](
            self["attn_conv_K"](aux))  # [B, n_head, C, T, Q]
        V = self["attn_norm_V"](self["attn_conv_V"](
            aux))  # [B, n_head, CV, T, Q], [B, 4, 32, T, 65]
        Q = Q.view(-1, *Q.shape[2:])  # [B*n_head, C, T, Q]
        K = K.view(-1, *K.shape[2:])  # [B*n_head, C, T, Q]
        V = V.view(-1, *V.shape[2:])  # [B*n_head, CV, T, Q]

        Q = Q.transpose(1, 2)
        Q = Q.flatten(start_dim=2)  # [B', T, C*Q]

        K = K.transpose(2, 3)
        K = K.contiguous().view([B * self.n_head, -1, aux_T])  # [B', C*Q, T]

        V = V.transpose(1, 2)  # [B', T, CV, Q]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, CV*Q]
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K) / (emb_dim**0.5)  # [B', T, T]
        attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, CV*Q]

        # V = V.reshape(old_shape)  # [B', T, C, Q]
        V = V.reshape([old_shape[0], old_T, old_shape[-2], old_shape[-1]])
        V = V.transpose(1, 2)  # [B', CV, T, Q]
        emb_dim = V.shape[1]

        batch = V.contiguous().view([B, self.n_head * emb_dim, old_T,
                                     old_Q])  # [B, emb_dim, T, Q])
        batch = self["attn_concat_proj"](batch)  # [B, emb_dim, T, Q])

        return batch


class LayerNormalization4DCF(nn.Module):

    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError(
                "Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) +
            self.eps)  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class AllHeadPReLULayerNormalization4DCF(nn.Module):

    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 3
        H, E, n_freqs = input_dimension  # 4, 8, 65
        param_size = [1, H, E, 1, n_freqs]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.act = nn.PReLU(num_parameters=H, init=0.25)
        self.eps = eps
        self.H = H
        self.E = E
        self.n_freqs = n_freqs

    def forward(self, x):
        assert x.ndim == 4
        B, _, T, _ = x.shape
        x = x.view([B, self.H, self.E, T, self.n_freqs])
        x = self.act(x)  # [B,H,E,T,F]
        stat_dim = (2, 4)
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,H,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) +
            self.eps)  # [B,H,1,T,1]
        x = ((x - mu_) / std_) * self.gamma + self.beta  # [B,H,E,T,F]
        return x
