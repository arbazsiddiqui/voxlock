"""Export the model as two ONNX graphs for onnxruntime-web, with no complex tensors and no STFT ops.

  enroll.onnx  voice sample (1, Te) -> enroll_mag (1, 257, Fe), context (1, C, Fc)        run once per call
  chunk.onnx   frames of the padded call + LSTM state -> windowed output frames + new state   run per chunk

Only the six time-direction LSTMs carry anything across frames (every norm, band LSTM, attention and mask works on one
frame at a time), so chunked streaming with the LSTM state carried over equals the full-file forward pass.

Caller side (JS or Python, see stream_reference below): reflect-pad the call by 256 samples at both ends (torch.stft
center=True), feed hops of K frames (K*128 + 384 samples, consecutive chunks overlap by 384), overlap-add the returned
frames at a 128-sample hop, divide by the summed squared Hann window, drop the first 256 samples, cut to the call length.

    python export_onnx.py --ckpt ckpt --out onnx/
"""
import argparse, math, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent / "wesep"))
import wesep
from torchaudio.compliance.kaldi import get_mel_banks

N_FFT, HOP, NF = 512, 128, 257

def dft_basis(n_in, n_fft, window):
    k = torch.arange(NF, dtype=torch.float64)[:, None]
    n = torch.arange(n_in, dtype=torch.float64)[None]
    ang = 2 * math.pi * k * n / n_fft
    w = window.double()[None]
    return (torch.cos(ang) * w).float(), (-torch.sin(ang) * w).float()  # (NF, n_in) each

class ConvSTFT(nn.Module):
    """torch.stft(n_fft=512, hop=128, hann periodic) without centre padding: (B, T) -> re, im (B, 257, frames)."""
    def __init__(self):
        super().__init__()
        cr, ci = dft_basis(N_FFT, N_FFT, torch.hann_window(N_FFT))
        self.register_buffer("w", torch.cat([cr, ci])[:, None])

    def forward(self, x):
        y = F.conv1d(x[:, None], self.w, stride=HOP)
        return y[:, :NF], y[:, NF:]

class FrameISTFT(nn.Module):
    """Per-frame inverse real DFT times the Hann synthesis window: re, im (B, 257, K) -> (B, K, 512)."""
    def __init__(self):
        super().__init__()
        k = torch.arange(NF, dtype=torch.float64)[:, None]
        n = torch.arange(N_FFT, dtype=torch.float64)[None]
        c = torch.full((NF, 1), 2.0, dtype=torch.float64); c[0] = c[-1] = 1.0
        ang = 2 * math.pi * k * n / N_FFT
        w = torch.hann_window(N_FFT).double()[None]
        self.register_buffer("br", (c * torch.cos(ang) / N_FFT * w).float())
        si = -c * torch.sin(ang) / N_FFT * w
        si[0] = 0; si[-1] = 0  # irfft ignores the imaginary part of DC and Nyquist
        self.register_buffer("bi", si.float())

    def forward(self, re, im):
        return torch.matmul(re.transpose(1, 2), self.br) + torch.matmul(im.transpose(1, 2), self.bi)

class KaldiFbank(nn.Module):
    """torchaudio.compliance.kaldi.fbank as the model calls it (x32768, 25/10 ms, hamming, 80 mel, no dither) + CMN."""
    def __init__(self):
        super().__init__()
        L = 400
        self.register_buffer("eye", torch.eye(L)[:, None])
        win = torch.hamming_window(L, periodic=False, alpha=0.54, beta=0.46)
        cr, ci = dft_basis(L, N_FFT, win)
        self.register_buffer("cr", cr.T.contiguous()); self.register_buffer("ci", ci.T.contiguous())
        mel, _ = get_mel_banks(80, N_FFT, 16000.0, 20.0, 0.0, 100.0, -500.0, 1.0)
        self.register_buffer("mel", F.pad(mel, (0, 1)).T.contiguous())  # (257, 80)

    def forward(self, x):
        fr = F.conv1d(x[:, None] * 32768.0, self.eye, stride=160).transpose(1, 2)  # (B, M, 400)
        fr = fr - fr.mean(-1, keepdim=True)
        prev = torch.cat([fr[..., :1], fr[..., :-1]], -1)
        fr = fr - 0.97 * prev
        p = torch.matmul(fr, self.cr) ** 2 + torch.matmul(fr, self.ci) ** 2
        m = torch.log(torch.clamp(torch.matmul(p, self.mel), min=torch.finfo(torch.float32).eps))
        return m - m.mean(1, keepdim=True)  # (B, M, 80)

class Enroll(nn.Module):
    def __init__(self, tse):
        super().__init__()
        self.stft, self.fbank = ConvSTFT(), KaldiFbank()
        self.spk = tse.spk_ft.context.encoder.spk_model

    def forward(self, enroll):
        x = F.pad(enroll[:, None], (256, 256), mode="reflect")[:, 0]
        re, im = self.stft(x)
        mag = torch.sqrt(re ** 2 + im ** 2)
        emb = self.spk._get_frame_level_feat(self.fbank(enroll))
        if isinstance(emb, tuple):
            emb = emb[-1]
        return mag, emb

class Chunk(nn.Module):
    def __init__(self, tse):
        super().__init__()
        self.t = tse
        self.stft, self.istft = ConvSTFT(), FrameISTFT()
        sep = tse.sep_model
        self.bw = sep.band_split.band_width
        self.nband = sep.nband

    def forward(self, wav, enroll_mag, context, h, c):
        """wav (1, K*128+384); h, c (6, nband, 256) -> frames (1, K, 512), h, c."""
        t, sep = self.t, self.t.sep_model
        re, im = self.stft(wav)                                      # (1, 257, K)
        mag = torch.sqrt(re ** 2 + im ** 2)
        spec = torch.stack([re, im], 1)                              # (1, 2, 257, K)
        tfmap = t.spk_ft.tfmap.compute(enroll_mag, mag)              # (1, 257, K)
        spec = torch.cat([spec, tfmap[:, None]], 1)                  # (1, 3, 257, K)
        sub, sre, sim, i0 = [], [], [], 0
        for bw in self.bw:
            sub.append(spec[:, :, i0: i0 + bw].contiguous()); sre.append(re[:, i0: i0 + bw]); sim.append(im[:, i0: i0 + bw]); i0 += bw
        feat = sep.subband_norm(sub)                                 # (1, nband, 128, K)
        feat = t.spk_ft.context.post(feat, context)
        B, nb, fd, K = feat.shape
        x = feat.reshape(B, nb * fd, K)
        hs, cs = [], []
        for j, blk in enumerate(sep.separator.separation):
            r = blk.band_rnn
            inp = x.view(B * nb, fd, K)
            o, (hj, cj) = r.rnn(r.norm(inp).transpose(1, 2), (h[j: j + 1], c[j: j + 1]))
            hs.append(hj); cs.append(cj)
            o = r.proj(o)                                            # (B*nb, K, fd)
            band = (inp + o.transpose(1, 2)).view(B, nb, fd, K)
            band = band.permute(0, 3, 2, 1).reshape(B * K, fd, nb)
            x = blk.band_comm(band).view(B, K, fd, nb).permute(0, 3, 2, 1).reshape(B, nb * fd, K)
        out = x.view(B, nb, fd, K)
        er, ei = [], []
        for i, mf in enumerate(sep.band_masker.mask):
            o = mf(out[:, i]).view(B, 1, 2, 2, self.bw[i], K)
            m = o[:, :, 0] * torch.sigmoid(o[:, :, 1])
            mr, mi = m[:, 0, 0], m[:, 0, 1]
            er.append(sre[i] * mr - sim[i] * mi); ei.append(sre[i] * mi + sim[i] * mr)
        frames = self.istft(torch.cat(er, 1), torch.cat(ei, 1))     # (1, K, 512)
        return frames, torch.cat(hs), torch.cat(cs)

def hann_sq():
    return (torch.hann_window(N_FFT) ** 2).numpy()

def stream_reference(enroll_fn, chunk_fn, mix, enr, K=100, nband=32, hid=256):
    """The exact caller-side loop the browser runs, in numpy. Returns the isolated signal, same length as mix."""
    mag, ctx = enroll_fn(enr[None].astype(np.float32))
    x = np.pad(mix, (256, 256), mode="reflect").astype(np.float32)
    nfr = 1 + (len(x) - N_FFT) // HOP
    pad = (nfr - 1) * HOP + N_FFT - len(x)
    x = np.pad(x, (0, max(0, pad)))
    out = np.zeros((nfr - 1) * HOP + N_FFT, np.float32)
    env = np.zeros_like(out)
    w2 = hann_sq()
    h = np.zeros((6, nband, hid), np.float32); c = np.zeros_like(h)
    for f0 in range(0, nfr, K):
        k = min(K, nfr - f0)
        seg = x[f0 * HOP: (f0 + k - 1) * HOP + N_FFT]
        fr, h, c = chunk_fn(seg[None], mag, ctx, h, c)
        for j in range(k):
            s = (f0 + j) * HOP
            out[s: s + N_FFT] += fr[0, j]; env[s: s + N_FFT] += w2
    y = out / np.where(env > 1e-11, env, 1.0)
    return y[256: 256 + len(mix)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", default="onnx")
    a = ap.parse_args()
    ex = wesep.load_model_local(a.ckpt)
    tse = ex.model.eval().cpu()
    tse.spk_ft.context.fbank.dither = 0.0
    Path(a.out).mkdir(exist_ok=True)
    en, ch = Enroll(tse).eval(), Chunk(tse).eval()
    nb = tse.sep_model.nband
    with torch.no_grad():
        e = torch.randn(1, 48000) * 0.1
        mag, ctx = en(e)
        torch.onnx.export(en, (e,), f"{a.out}/enroll.onnx", input_names=["enroll"], output_names=["enroll_mag", "context"],
                          dynamic_axes={"enroll": {1: "Te"}, "enroll_mag": {2: "Fe"}, "context": {2: "Fc"}}, opset_version=17, dynamo=False)
        K = 50
        w = torch.randn(1, K * HOP + 384) * 0.1
        h = torch.zeros(6, nb, 256); c = torch.zeros(6, nb, 256)
        torch.onnx.export(ch, (w, mag, ctx, h, c), f"{a.out}/chunk.onnx", input_names=["wav", "enroll_mag", "context", "h", "c"],
                          output_names=["frames", "h_out", "c_out"],
                          dynamic_axes={"wav": {1: "S"}, "enroll_mag": {2: "Fe"}, "context": {2: "Fc"}, "frames": {1: "K"}},
                          opset_version=17, dynamo=False)
    print("exported", nb, "bands")
