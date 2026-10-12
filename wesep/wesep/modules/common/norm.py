import numbers

import torch
import torch.nn as nn


def select_norm(norm, dim, eps=1e-5, group=1):
    """
    Build normalize layer
    LN cost more memory than BN
    """
    if norm not in ["cLN", "LN", "gLN", "GN", "BN"]:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, eps, elementwise_affine=True)
    elif norm == "LN":
        # dim can be int or tuple
        return nn.LayerNorm(dim, eps, elementwise_affine=True)
    elif norm == "GN":
        return nn.GroupNorm(group, dim, eps)
    elif norm == "BN":
        return nn.BatchNorm1d(dim, eps)
    else:
        return GlobalChannelLayerNorm(dim, eps, elementwise_affine=True)


class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: N x C x T
        """
        x = torch.transpose(x, 1, -1)
        x = super().forward(x)
        x = torch.transpose(x, 1, -1)
        return x


class GlobalChannelLayerNorm(nn.Module):
    """
    Calculate Global Layer Normalization
    dim: (int or list or torch.Size) –
         input shape from an expected input of size
    eps: a value added to the denominator for numerical stability.
    elementwise_affine: a boolean value that when set to True,
        this module has learnable per-element affine parameters
        initialized to ones (for weights) and zeros (for biases).
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalChannelLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.dim, 1))
            self.bias = nn.Parameter(torch.zeros(self.dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        # x = N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x L
        # gln: mean,var N x 1 x 1
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))

        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean)**2, (1, 2), keepdim=True)
        # N x C x L
        if self.elementwise_affine:
            x = (self.weight * (x - mean) / torch.sqrt(var + self.eps) +
                 self.bias)
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class ConditionalLayerNorm(nn.Module):
    """
    https://github.com/HuangZiliAndy/fairseq/blob/multispk/fairseq/models/wavlm/WavLM.py#L1160
    """

    def __init__(self,
                 normalized_shape,
                 embed_dim,
                 modulate_bias=False,
                 eps=1e-5):
        super(ConditionalLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape, )
        self.normalized_shape = tuple(normalized_shape)

        self.embed_dim = embed_dim
        self.eps = eps

        self.weight = nn.Parameter(torch.empty(*normalized_shape))
        self.bias = nn.Parameter(torch.empty(*normalized_shape))
        assert len(normalized_shape) == 1
        self.ln_weight_modulation = FiLM(normalized_shape[0], embed_dim)
        self.modulate_bias = modulate_bias
        if self.modulate_bias:
            self.ln_bias_modulation = FiLM(normalized_shape[0], embed_dim)
        else:
            self.ln_bias_modulation = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input, embed):
        mean = torch.mean(input, -1, keepdim=True)
        var = torch.var(input, -1, unbiased=False, keepdim=True)
        weight = self.ln_weight_modulation(
            embed, self.weight.expand(embed.size(0), -1))
        if self.ln_bias_modulation is None:
            bias = self.bias
        else:
            bias = self.ln_bias_modulation(embed,
                                           self.bias.expand(embed.size(0), -1))
        res = (input - mean) / torch.sqrt(var + self.eps) * weight + bias
        return res

    def extra_repr(self):
        return "{normalized_shape}, {embed_dim}, \
            modulate_bias={modulate_bias}, eps={eps}".format(**self.__dict__)
