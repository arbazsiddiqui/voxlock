# Copyright (c) 2025 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
#
# Description: wesep v2 network component.

from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

from wesep.modules.speaker.spk_frontend import SpeakerFrontend
from wesep.modules.separator.bsrnn import BSRNN
from wesep.modules.common.deep_update import deep_update


class TSE_BSRNN_SPK(nn.Module):

    def __init__(self, config):
        super().__init__()

        # ===== Merge configs =====
        sep_configs = dict(
            sr=16000,
            win=512,
            stride=128,
            feature_dim=128,
            num_repeat=6,
            causal=False,
            nspk=1,  # For Separation (multiple output)
            spec_dim=2,  # For TSE feature, used in self.subband_norm
        )
        sep_configs = {**sep_configs, **config['separator']}
        spk_configs = {
            "features": {
                "listen": {
                    "enabled": False,
                    "win": sep_configs["win"],
                    "hop": sep_configs["stride"],
                },
                "usef": {
                    "enabled": False,
                    "causal": sep_configs["causal"],
                    "enc_dim": sep_configs["win"] // 2 + 1,
                    "emb_dim": sep_configs["feature_dim"] // 2,
                },
                "tfmap": {
                    "enabled": False
                },
                "context": {
                    "enabled": False,
                    "mix_dim": sep_configs["feature_dim"],
                    "atten_dim": sep_configs["feature_dim"]
                },
                "spkemb": {
                    "enabled": False,
                    "mix_dim": sep_configs["feature_dim"]
                },
            },
            "speaker_model": {
                "fbank": {
                    "sample_rate": sep_configs["sr"]
                },
            },
        }
        self.spk_configs = deep_update(spk_configs, config['speaker'])
        # ===== Separator Loading =====
        if self.spk_configs["features"]["usef"]["enabled"]:
            sep_configs["spec_dim"] = self.spk_configs["features"]["usef"][
                "emb_dim"] * 2
        if self.spk_configs["features"]["tfmap"]["enabled"]:
            sep_configs["spec_dim"] = sep_configs["spec_dim"] + 1  #
        self.sep_model = BSRNN(**sep_configs)
        # ===== Speaker Loading =====
        if self.spk_configs["features"]["context"]["enabled"]:
            self.spk_configs["features"]["context"][
                "band"] = self.sep_model.nband  #
        self.spk_ft = SpeakerFrontend(self.spk_configs)

    def forward(self, mix, enroll):
        """
        Args:
            mix:  Tensor [B, 1, T]
            enroll: list[Tensor]
                each Tensor: [B, 1, T]
        """

        if isinstance(enroll, (list, tuple)):
            enroll = enroll[0]
        mix = mix.squeeze(1)
        enroll = enroll.squeeze(1)

        # input shape: (B, T)
        mix_dims = mix.dim()
        assert mix_dims == 2, "Only support 2D Input"

        ##### Cue of the target speaker
        wav_enroll = enroll
        ###### Extraction with speaker cue
        batch_size, nsamples = mix.shape
        wav_mix = mix
        ###########################################################
        # C0. Feature: listen
        if self.spk_configs['features']['listen']['enabled']:
            # C0.1 Prepend the enroll to the mix in the beginning
            wav_mix = self.spk_ft.listen.compute(wav_enroll,
                                                 wav_mix)  # (B, T_e + T_s + T)
        ###########################################################
        # S1. Convert into frequency-domain
        spec = self.sep_model.stft(wav_mix)[-1]
        # S2. Concat real and imag, split to subbands
        spec_RI = torch.stack([spec.real, spec.imag], 1)  # (B, 2, F, T)
        ###########################################################
        # C1. Feature: usef
        if self.spk_configs['features']['usef']['enabled']:
            # C1.1 Generate the USEF feature
            enroll_spec = self.sep_model.stft(wav_enroll)[
                -1]  # (B, F, T_e) complex
            enroll_spec = torch.stack([enroll_spec.real, enroll_spec.imag],
                                      1)  # (B, 2, F, T)
            enroll_usef, mix_usef = self.spk_ft.usef.compute(
                enroll_spec, spec_RI)  # (B, embed_dim, F, T)
            # C1.2 Concate the USEF feature to the mix_repr's spec
            spec_RI = self.spk_ft.usef.post(
                mix_usef, enroll_usef)  # (B, embed_dim*2, F, T)
        # C2. Feature: tfmap
        if self.spk_configs['features']['tfmap']['enabled']:
            # C2.1 Generate the TF-Map feature
            enroll_mag = self.sep_model.stft(wav_enroll)[0]  # (B, F, T_e)
            enroll_tfmap = self.spk_ft.tfmap.compute(
                enroll_mag, torch.abs(spec))  # (B, F, T)
            # C2.2 Concate the TF-Map feature to the mix_repr's spec
            spec_RI = self.spk_ft.tfmap.post(
                spec_RI, enroll_tfmap.unsqueeze(1))  # (B, 3, F, T)
        ###########################################################
        subband_spec = self.sep_model.band_split(
            spec_RI)  # list of (B, 2/3/2*usef.emb_dim, BW, T)
        subband_mix_spec = self.sep_model.band_split(
            spec)  # list of (B, BW, T) complex
        # S3. Normalization and bottleneck
        subband_feature = self.sep_model.subband_norm(
            subband_spec)  # (B, nband, feat, T)
        ###########################################################
        # C3. Feature: context
        if self.spk_configs['features']['context']['enabled']:
            # C3.1 Generate the frame-level speaker embeddings
            enroll_context = self.spk_ft.context.compute(
                wav_enroll)  # (B, F_e, T_e)
            # C3.2 Fuse the frame-level speaker embeddings into the mix_repr
            subband_feature = self.spk_ft.context.post(
                subband_feature, enroll_context)  # (B, nband, feat, T)
        # C4. Feature: spkemb
        if self.spk_configs['features']['spkemb']['enabled']:
            # C4.1 Generate the speaker embedding
            enroll_emb = self.spk_ft.spkemb.compute(wav_enroll)  # (B, F_e)
            # C4.2 Fuse the speaker embeeding into the mix_repr
            enroll_emb = enroll_emb.unsqueeze(1).unsqueeze(3)  # (B, 1, F_e, 1)
            subband_feature = self.spk_ft.spkemb.post(
                subband_feature, enroll_emb)  # (B, nband, feat, T)
        ###########################################################
        # S4. Separation
        sep_output = self.sep_model.separator(
            subband_feature)  # (B, nband, feat, T)
        # S5. Complex Mask
        est_spec_RI = self.sep_model.band_masker(
            sep_output, subband_mix_spec)  # (B, 2, S, F, T)
        est_complex = torch.complex(est_spec_RI[:, 0],
                                    est_spec_RI[:, 1])  # (B, S, F, T)
        # S6. Back into waveform
        s = self.sep_model.istft(est_complex)  # (B, S, T)
        ###########################################################
        # C0. Feature: listen
        if self.spk_configs['features']['listen']['enabled']:
            # C0.2 Prepend the enroll to the mix in the beginning
            s = self.spk_ft.listen.post(s)  # (B, T)
        ###########################################################
        return s


def check_causal(model):
    input = torch.randn(1, 16000 * 8).clamp_(-1, 1)
    enroll = torch.randn(1, 16000 * 2).clamp_(-1, 1)
    fs = 16000
    model = model.eval()
    with torch.no_grad():
        out1 = model(input, enroll)
        for i in range(fs * 1, fs * 4, fs):
            inputs2 = input.clone()
            inputs2[..., i:] = 1 + torch.rand_like(inputs2[..., i:])
            out2 = model(inputs2, enroll)
            print((((out1[0] - out2[0]).abs() > 1e-8).float().argmax()) / fs)
            print((((inputs2 - input).abs() > 1e-8).float().argmax()) / fs)


if __name__ == "__main__":
    from thop import profile, clever_format

    config = dict()
    config['separator'] = dict(
        sr=16000,
        win=512,
        stride=128,
        feature_dim=128,
        num_repeat=6,
        causal=True,
        nspk=1,
    )
    config['speaker'] = {
        "features": {
            "listen": {
                "enabled": True
            },
            "usef": {
                "enabled": True
            },
            "tfmap": {
                "enabled": True
            },
            "context": {
                "enabled": True
            },
            "spkemb": {
                "enabled": True
            },
        }
    }
    model = TSE_BSRNN_SPK(config)
    s = 0
    for param in model.parameters():
        s += np.product(param.size())
    print("# of parameters: " + str(s / 1024.0 / 1024.0))
    mix = torch.randn(4, 32000)
    enroll = torch.randn(4, 31235)
    model = model.eval()
    with torch.no_grad():
        output = model(mix, enroll)
    print(output.shape)

    check_causal(model)
    exit()

    macs, params = profile(model, inputs=(x, spk_embeddings))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)
