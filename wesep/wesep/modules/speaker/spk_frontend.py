# Copyright (c) 2025 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
#
# Description: wesep v2 network component.

import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.common.deep_update import deep_update
from wesep.modules.common.norm import select_norm
from wesep.modules.fusion.speech import SpeakerFuseLayer, CrossFuse
from wesep.modules.speaker.encoder import Fbank_kaldi, SpeakerEncoder
from wesep.modules.speaker.usef import USEF_attentionblock


class BaseSpeakerFeature(nn.Module):

    def compute(self, enroll, mix=None):
        """返回该特征的结果"""
        raise NotImplementedError

    def post(self, mix_repr, feat_repr):
        """返回融合后的 mix_repr"""
        raise NotImplementedError


class ListenFeature(BaseSpeakerFeature):

    def __init__(self, config):
        super().__init__()
        self.silence_len = config["glue"]
        self.win = config["win"]
        self.hop = config["hop"]
        self.offset = 0

    def compute(self, enroll, mix):
        """
        enroll: (B, T_e)
        mix:    (B, T)
        return:
            x: (B, T_e + T_s + pad + T)
        """
        B, _ = mix.shape
        device = mix.device
        dtype = mix.dtype

        enroll_len = enroll.shape[-1]
        silence_len = self.silence_len

        # ---- silence ----
        if silence_len > 0:
            silence = torch.zeros(B, silence_len, device=device, dtype=dtype)
            prefix = torch.cat([enroll, silence], dim=-1)
        else:
            prefix = enroll

        prefix_len = prefix.shape[-1]

        # For (prefix_len - win) % hop == 0
        pad = (self.hop - ((prefix_len - self.win) % self.hop)) % self.hop

        if pad > 0:
            pad_tensor = torch.zeros(B, pad, device=device, dtype=dtype)
            prefix = torch.cat([prefix, pad_tensor], dim=-1)

        self.offset = prefix.shape[-1]
        assert self.offset % self.hop == 0

        x = torch.cat([prefix, mix], dim=-1)
        return x

    def post(self, mix_repr, mix_len=None):
        """
        mix_repr: (..., T_total)
        mix_len:  length of mix
        """
        out = mix_repr[..., self.offset:]
        if mix_len is not None:
            out = out[..., :mix_len]
        return out


class TFMapFeature(BaseSpeakerFeature):

    def __init__(self, config):
        super().__init__()
        ## Only support TF_Map Spec

    def compute(self, enroll, mix):
        """
        enroll: (B, F, T) magnitude of spec
        mix: (B, F, T)  magnitude of spec
        return:
            tf_map: (B, F, T)
        """
        mix_mag = F.normalize(mix, p=2, dim=1)
        enroll_mag = F.normalize(enroll, p=2, dim=1)

        mix_mag = mix_mag.permute(0, 2, 1).contiguous()
        att_scores = torch.matmul(mix_mag, enroll_mag)
        att_weights = F.softmax(att_scores, dim=-1)
        enroll_mag = enroll_mag.permute(0, 2, 1).contiguous()
        tf_map = torch.matmul(att_weights, enroll_mag)
        tf_map = tf_map.permute(0, 2, 1).contiguous()

        tf_map = tf_map / tf_map.norm(dim=1, keepdim=True)
        # Recover the energy of estimated tfmap feature
        tf_map = (torch.sum(mix * tf_map, dim=1, keepdim=True) * tf_map)
        # Another kind of nomalization for tf_map feature
        # tf_map = tf_map * mix.norm(dim=1, keepdim=True)
        return tf_map

    def post(self, mix_repr, feat_repr):
        """
        mix_repr: (B, 2, F, T)
        feat_repr: (B, 1, F, T)
        return:
            (B, 3, F, T)
        """
        return torch.cat([mix_repr, feat_repr], dim=1)


class UsefFeature(BaseSpeakerFeature):

    def __init__(self, config, eps=torch.finfo(torch.float32).eps):
        super().__init__()

        self.causal = config["causal"]
        norm_type = "LN" if self.causal else "GN"

        ks = (config["t_ksize"], 3)
        padding = (config["t_ksize"] // 2, 1)

        if not self.causal:
            # === non-causal: shared encoder ===
            self.usef_con2v = nn.Sequential(
                select_norm(norm_type, config["spec_dim"], eps),
                nn.Conv2d(config["spec_dim"],
                          config["emb_dim"],
                          ks,
                          padding=padding),
            )
        else:
            self.usef_con2v = nn.Sequential(
                select_norm(norm_type, (config["spec_dim"], config["enc_dim"]),
                            eps),
                nn.Conv2d(config["spec_dim"],
                          config["emb_dim"],
                          ks,
                          padding=padding),
            )
            # === causal: individual encoder + norm ===
            # self.mix_encoder = nn.Sequential(
            #     select_norm("cLN", config["spec_dim"], eps),
            #     nn.Conv2d(config["spec_dim"], config["emb_dim"], ks, padding=padding),
            # )
            # self.enroll_encoder = nn.Sequential(
            #     select_norm("GN", config["spec_dim"], eps),
            #     nn.Conv2d(config["spec_dim"], config["emb_dim"], ks, padding=padding),
            # )
        self.usef_att = USEF_attentionblock(
            emb_dim=config["emb_dim"],
            n_freqs=config["enc_dim"],
            n_head=config["n_head"],
            approx_qk_dim=config["approx_qk_dim"],
        )

    def compute(self, enroll, mix):
        """
        enroll: (B, 2, F, T)
        mix: (B, 2, F, T)
        return:
            enroll: (B, 128, F, T)
            mix: (B, 128, F, T)
        """

        mix = mix.permute(0, 1, 3, 2).contiguous()  # B, 2, T, F
        enroll = enroll.permute(0, 1, 3, 2).contiguous()  # B, 2, T, F

        if not self.causal:
            mix = self.usef_con2v(mix)  # B, 128, T, F
            enroll = self.usef_con2v(enroll)
        else:
            mix = mix.permute(0, 2, 1, 3).contiguous()  # B, T, 2, F
            mix = self.usef_con2v[0](mix)  # LN over 2, F
            mix = mix.permute(0, 2, 1, 3).contiguous()  # B, 2, T, F
            mix = self.usef_con2v[1](mix)  # Conv2d

            enroll = enroll.permute(0, 2, 1, 3).contiguous()
            enroll = self.usef_con2v[0](enroll)
            enroll = enroll.permute(0, 2, 1, 3).contiguous()
            enroll = self.usef_con2v[1](enroll)

        enroll = self.usef_att(mix, enroll)  # B, 128, T, F

        enroll = enroll.permute(0, 1, 3, 2).contiguous()  # B, 128, F, T
        mix = mix.permute(0, 1, 3, 2).contiguous()  # B, 128, F, T

        return enroll, mix

    def post(self, mix_repr, feat_repr):
        return torch.cat([mix_repr, feat_repr], dim=1)


class ContextFeature(BaseSpeakerFeature):

    def __init__(
        self,
        conf_context,
        conf_spk=None,
        fbank=None,
        encoder=None,
    ):
        super().__init__()
        if conf_context["speaker_model"]:
            self.fbank = Fbank_kaldi(**conf_context["speaker_model"]['fbank'])
            self.encoder = SpeakerEncoder(
                conf_context["speaker_model"]['speaker_encoder'])
        else:
            self.fbank = fbank
            self.encoder = encoder
        self.attenFuse = CrossFuse(
            embed_dim=conf_context['embed_dim'],
            atten_dim=conf_context['atten_dim'],
            mix_dim=conf_context['mix_dim'],
            num_heads=conf_context['num_heads'],  # 2 or 4
            nband=conf_context[
                'band'],  # kdim=spk_emb_frame_dim, vdim=spk_emb_frame_dim,
            batch_first=True)
        self.fusionLayer = SpeakerFuseLayer(
            embed_dim=conf_context['atten_dim'],
            feat_dim=conf_context['mix_dim'],
            fuse_type=conf_context['fusion'],
        )

    def compute(self, enroll, mix=None):
        """
        enroll: (B, T)
        return:
            emb: (B, F, T)
        """
        fb = self.fbank(enroll)
        emb = self.encoder.spk_model._get_frame_level_feat(fb)
        if isinstance(emb, tuple):
            emb = emb[-1]  # (B, F_e, T_e)
        return emb

    def post(self, mix_repr, emb):
        """
        mix_repr: (B, F, T) or (B, band, F, T)
        emb: (B, F_e, T_e)
        return:
            mix_repr: (B, F, T) or (B, band, F, T)
        """
        emb = self.attenFuse(mix_repr, emb)  # (B, F, T) or (B, band, F, T)
        mix_repr = self.fusionLayer(mix_repr, emb)
        return mix_repr


class SpeakerEmbFeature(BaseSpeakerFeature):

    def __init__(
        self,
        conf_emb,
        conf_spk,
        fbank=None,
        encoder=None,
    ):
        super().__init__()
        if conf_emb["speaker_model"]:
            self.fbank = Fbank_kaldi(**conf_emb["speaker_model"]['fbank'])
            self.encoder = SpeakerEncoder(
                conf_emb["speaker_model"]['speaker_encoder'])
            embed_dim = conf_emb['speaker_encoder']['spk_args']['embed_dim']
        else:
            self.fbank = fbank
            self.encoder = encoder
            embed_dim = conf_spk['speaker_encoder']['spk_args']['embed_dim']
        self.fusionLayer = SpeakerFuseLayer(
            embed_dim=embed_dim,
            feat_dim=conf_emb['mix_dim'],
            fuse_type=conf_emb['fusion'],
        )

    def compute(self, enroll, mix=None):
        """
        enroll: (B, T)
        return:
            emb: (B, F)
        """
        fb = self.fbank(enroll)
        emb = self.encoder(fb)
        if isinstance(emb, tuple):
            emb = emb[-1]  # (B, F)
        # emb.unsqueeze(-1).expand(-1, -1, mix.size(-1))  # broadcast
        return emb

    def post(self, mix_repr, emb):
        return self.fusionLayer(mix_repr, emb)  # or concat, configurable


class SpeakerFrontend(nn.Module):

    def __init__(self, config):
        super().__init__()

        # ===== Default Config =====
        DEFAULT_SPK_CONFIG = {
            "features": {
                # ---- Features without Speaker Encoder
                "listen": {
                    "enabled": False,
                    "glue": 512,
                    "win": 512,
                    "hop": 128,
                },
                "usef": {
                    "enabled": False,
                    "causal": False,
                    "spec_dim": 2,
                    "emb_dim": 128,  # 64
                    "enc_dim": 65,
                    "approx_qk_dim": 512,
                    "n_head": 4,
                    "t_ksize": 3,
                    # "fusion": 'concat',
                },
                "tfmap": {
                    "enabled": False,
                    "type": 'spec',  # 'spec', 'emb'
                    # "speaker_model": None,
                    # "fusion": 'concat',
                },

                # ---- Features with Speaker Encoder, sharing fbank + speaker encoder）
                "context": {
                    "enabled": False,
                    "speaker_model": None,
                    "embed_dim":
                    512,  # Need check the speaker encoder in wespeaker
                    "atten_dim": 128,  # dim in cross-attention
                    "num_heads": 2,
                    "fusion": "multiply",  # add | concat | multiply
                    "mix_dim": 128,
                    "band": 1,
                },
                "spkemb": {
                    "enabled": False,
                    "speaker_model": None,
                    "fusion": "multiply",  # add | concat | multiply
                    "mix_dim": 128,
                },
            },

            # ===== Sharing coefficients, for context + spkemb =====
            "speaker_model": {
                # ---- Fbank
                "fbank": {
                    "num_mel_bins": 80,
                    "frame_shift": 10,
                    "frame_length": 25,
                    "dither": 0.0,
                    "sample_rate": 16000,  # sampling rate
                },

                # ---- Speaker Encoder
                "speaker_encoder": {
                    "model": 'ECAPA_TDNN_GLOB_c512',
                    "pretrained":
                    './wespeaker_models/voxceleb_ECAPA512/avg_model.pt',
                    "spk_args": {
                        "embed_dim": 192,  # speaker embedding
                        "feat_dim": 80,  # num_mel_bins
                        "pooling_func": 'ASTP',
                    },
                },
            },
        }
        self.config = deep_update(DEFAULT_SPK_CONFIG, config)

        feats = self.config['features']
        if feats['context']['enabled'] or feats['spkemb']['enabled']:
            self.fbank = Fbank_kaldi(**self.config['speaker_model']['fbank'])
            self.encoder = SpeakerEncoder(
                self.config['speaker_model']['speaker_encoder'])

        if feats['listen']['enabled']:
            self.listen = ListenFeature(feats['listen'])

        if feats['usef']['enabled']:
            self.usef = UsefFeature(feats['usef'])

        if feats['tfmap']['enabled']:
            self.tfmap = TFMapFeature(feats['tfmap'])

        if feats['context']['enabled']:
            self.context = ContextFeature(
                feats['context'],
                self.config['speaker_model'],
                fbank=self.fbank,
                encoder=self.encoder,
            )

        if feats['spkemb']['enabled']:
            self.spkemb = SpeakerEmbFeature(
                feats['spkemb'],
                self.config['speaker_model'],
                fbank=self.fbank,
                encoder=self.encoder,
            )

    def compute_all(self, enroll, mix=None):
        out = {}
        for name, module in self.features.items():
            out[name] = module.compute(enroll, mix)
        return out


if __name__ == "__main__":

    model = SpeakerFrontend()
