from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.fusion.film import FiLM


class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            "flipped_filter",
            torch.FloatTensor([-self.coef, 1.0]).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), "reflect")
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class SpeakerTransform(nn.Module):

    def __init__(self, embed_dim=256, num_layers=3, hid_dim=128):
        """
        Transform the pretrained speaker embeddings, keep the dimension
        :param embed_dim:
        :param num_layers:
        :param hid_dim:
        :return:
        """
        super(SpeakerTransform, self).__init__()
        self.transforms = []
        self.transforms.append(nn.Conv1d(embed_dim, hid_dim, 1))
        for _ in range(num_layers - 2):
            self.transforms.append(nn.Conv1d(hid_dim, hid_dim, 1))
            self.transforms.append(nn.Tanh())
        self.transforms.append(nn.Conv1d(hid_dim, embed_dim, 1))
        self.transforms = nn.Sequential(*self.transforms)

    def forward(self, x):
        if len(x.size()) == 2:
            return self.transforms(x.unsqueeze(-1)).squeeze(-1)
        else:
            return self.transforms(x)


class LinearLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(LinearLayer, self).__init__()

        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x, dummy: Optional[torch.Tensor] = None):
        return self.linear(x)


class SpeakerFuseLayer(nn.Module):

    def __init__(self, embed_dim=256, feat_dim=512, fuse_type="concat"):
        super(SpeakerFuseLayer, self).__init__()
        assert fuse_type in ["concat", "additive", "multiply", "FiLM", "None"]

        self.fuse_type = fuse_type
        if fuse_type == "concat":
            self.fc = LinearLayer(embed_dim + feat_dim, feat_dim)
        elif fuse_type == "additive":
            self.fc = LinearLayer(embed_dim, feat_dim)
        elif fuse_type == "multiply":
            self.fc = LinearLayer(embed_dim, feat_dim)
        elif fuse_type == "FiLM":
            self.fc = FiLM(feat_dim, embed_dim)
        else:
            raise ValueError("Fuse type not defined.")

    def forward(self, x, embed):
        """

        :param x: batch x dimension x length
        :param embed: batch x dimension x 1
        :return:
        """
        if self.fuse_type == "concat":
            # For Conv
            if len(x.size()) == 3:
                embed_t = embed.expand(-1, -1, x.size(2))
                y = torch.cat([x, embed_t], 1)
                y = torch.transpose(y, 1, 2)
                x = torch.transpose(self.fc(y), 1, 2)
            else:
                # len(x.size() == 4
                embed_t = embed.expand(-1, x.size(1), -1, x.size(3))
                y = torch.cat([x, embed_t], 2)
                y = torch.transpose(y, 2, 3)
                x = torch.transpose(self.fc(y), 2, 3).contiguous()
                # print(x.size())
        elif self.fuse_type == "additive":
            if len(x.size()) == 3:
                embed_t = embed.expand(-1, -1, x.size(2))
                embed_t = torch.transpose(embed_t, 1, 2)
                x = x + torch.transpose(self.fc(embed_t), 1, 2)
            else:
                # len(x.size() == 4
                embed_t = embed.expand(-1, x.size(1), -1, x.size(3))
                embed_t = torch.transpose(embed_t, 2, 3)
                x = x + torch.transpose(self.fc(embed_t), 2, 3)
        elif self.fuse_type == "multiply":
            if len(x.size()) == 3:
                embed_t = embed.expand(-1, -1, x.size(2))
                embed_t = torch.transpose(embed_t, 1, 2)
                x = x * torch.transpose(self.fc(embed_t), 1, 2)
            else:
                # len(x.size() == 4
                embed_t = embed.expand(-1, x.size(1), -1, x.size(3))
                embed_t = torch.transpose(embed_t, 2, 3)
                x = x * torch.transpose(self.fc(embed_t), 2, 3)
        else:
            embed = embed.squeeze(-1)
            x = self.fc(embed, x)
        return x


class CrossFuse(nn.Module):

    def __init__(self,
                 embed_dim,
                 atten_dim,
                 mix_dim,
                 num_heads=1,
                 nband=1,
                 *args,
                 **kwargs):
        super().__init__()
        self.Linear = nn.ModuleList(
            [nn.Linear(embed_dim, atten_dim) for _ in range(nband)])
        if mix_dim != atten_dim:
            self.Linear_mix = nn.Linear(mix_dim, atten_dim)
        else:
            self.Linear_mix = nn.Identity()
        self.multihead_attn = nn.MultiheadAttention(atten_dim, num_heads,
                                                    *args, **kwargs)

    def forward(self, mix, emb):
        """
        mix: (B, F, T) or (B, band, F, T)
        emb: (B, F_e, T_e)
        return:
            spk_embeddings: (B, F, T) or (B, band, F, T)
        """
        if mix.dim() == 4:
            spk_embeddings = []
            for i in range(mix.shape[1]):
                query = mix[:,
                            i, :, :].squeeze(dim=1)  # (batch, feature, time)
                query = self.Linear_mix(query.transpose(1, 2))
                key = self.Linear[i](emb.transpose(1, 2))
                value = key
                x, _ = self.multihead_attn(query, key, value)
                spk_embeddings.append(x.transpose(1, 2))
            spk_embeddings = torch.stack(spk_embeddings, dim=1)
        elif mix.dim() == 3:
            query = self.Linear_mix(mix.transpose(1, 2))
            key = self.Linear[0](emb.transpose(1, 2))
            value = key
            x, _ = self.multihead_attn(query, key, value)
            spk_embeddings = x.transpose(1, 2)
        return spk_embeddings


def test_speaker_fuse():
    st = SpeakerTransform(embed_dim=256, num_layers=3, hid_dim=128)
    sfl = SpeakerFuseLayer(fuse_type="multiply")

    embeds = torch.rand(4, 256)
    encoder_output = torch.rand(4, 512, 1000)

    print(embeds.size())
    embeds = st(embeds)
    print(embeds.size())
    output = sfl(encoder_output, embeds)
    print(output.size())


if __name__ == "__main__":
    test_speaker_fuse()
