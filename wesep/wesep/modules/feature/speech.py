import torch
import torch.nn as nn


class STFT(nn.Module):

    def __init__(self, n_fft=256, hop_length=128, win_length=256):
        super(STFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, y):
        num_dims = y.dim()
        assert num_dims == 2 or num_dims == 3, "Only support 2D or 3D Input"

        batch_size = y.shape[0]
        num_samples = y.shape[-1]

        if num_dims == 3:
            y = y.reshape(-1, num_samples)  # [B * C ,T]

        complex_stft = torch.stft(
            y,
            self.n_fft,
            self.hop_length,
            self.win_length,
            window=torch.hann_window(self.win_length, device=y.device),
            return_complex=True,
        )
        _, num_freqs, num_frames = complex_stft.shape

        if num_dims == 3:
            complex_stft = complex_stft.reshape(batch_size, -1, num_freqs,
                                                num_frames)

        # print(complex_stft)

        mag = torch.abs(complex_stft)
        phase = torch.angle(complex_stft)
        real = complex_stft.real
        imag = complex_stft.imag
        return mag, phase, real, imag, complex_stft


class iSTFT(nn.Module):

    def __init__(self, n_fft=256, hop_length=128, win_length=256):
        super(iSTFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, features, input_type="complex", length=None):
        if input_type == "real_imag":
            # the feature is (real, imag) or [real, imag]
            assert isinstance(features, tuple) or isinstance(features, list)
            real, imag = features
            features = torch.complex(real, imag)
        elif input_type == "complex":
            assert torch.is_complex(
                features), "The input feature is not complex."
        elif input_type == "mag_phase":
            # the feature is (mag, phase) or [mag, phase]
            assert isinstance(features, tuple) or isinstance(features, list)
            mag, phase = features
            features = torch.complex(mag * torch.cos(phase),
                                     mag * torch.sin(phase))
        else:
            raise NotImplementedError(
                "Only 'real_imag', 'complex', and 'mag_phase' are supported.")

        num_dims = features.dim()
        assert num_dims == 3 or num_dims == 4, "Only support 3D or 4D Input"

        batch_size = features.shape[0]
        F = features.shape[-2]
        if num_dims == 4:
            nspk = features.shape[1]
            features = features.reshape(batch_size * nspk, F,
                                        -1)  # [B * S , F, T]

        wav_ouput = torch.istft(
            features,
            self.n_fft,
            self.hop_length,
            self.win_length,
            window=torch.hann_window(self.win_length, device=features.device),
            length=length,
        )

        if num_dims == 4:
            wav_ouput = wav_ouput.reshape(batch_size, nspk, -1)  # [B, S, T]
        return wav_ouput
