import torch.nn as nn


class FiLM(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer
    https://github.com/HuangZiliAndy/fairseq/blob/multispk/fairseq/models/wavlm/WavLM.py#L1160  # noqa
    """

    def __init__(self,
                 feat_size,
                 embed_size,
                 num_film_layers=1,
                 layer_norm=False):
        super(FiLM, self).__init__()
        self.feat_size = feat_size
        self.embed_size = embed_size
        self.num_film_layers = num_film_layers
        self.layer_norm = nn.LayerNorm(embed_size) if layer_norm else None
        gamma_fcs, beta_fcs = [], []
        for i in range(num_film_layers):
            if i == 0:
                gamma_fcs.append(nn.Linear(embed_size, feat_size))
                beta_fcs.append(nn.Linear(embed_size, feat_size))
            else:
                gamma_fcs.append(nn.Linear(feat_size, feat_size))
                beta_fcs.append(nn.Linear(feat_size, feat_size))
        self.gamma_fcs = nn.ModuleList(gamma_fcs)
        self.beta_fcs = nn.ModuleList(beta_fcs)
        self.init_weights()

    def init_weights(self):
        for i in range(self.num_film_layers):
            nn.init.zeros_(self.gamma_fcs[i].weight)
            nn.init.zeros_(self.gamma_fcs[i].bias)
            nn.init.zeros_(self.beta_fcs[i].weight)
            nn.init.zeros_(self.beta_fcs[i].bias)

    def forward(self, embed, x):
        gamma, beta = None, None
        for i in range(len(self.gamma_fcs)):
            if i == 0:
                gamma = self.gamma_fcs[i](embed)
                beta = self.beta_fcs[i](embed)
            else:
                gamma = self.gamma_fcs[i](gamma)
                beta = self.beta_fcs[i](beta)

        if len(gamma.shape) < len(x.shape):
            gamma = gamma.unsqueeze(-1).expand_as(x)
            beta = beta.unsqueeze(-1).expand_as(x)
        else:
            gamma = gamma.expand_as(x)
            beta = beta.expand_as(x)

        # print(gamma.size(), beta.size())
        x = (1 + gamma) * x + beta
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x
