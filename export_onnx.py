"""Export the model as two ONNX graphs for onnxruntime-web, with no complex tensors and no STFT ops.

  enroll.onnx  voice sample (1, Te) -> enroll_n (257, Fe), attention keys k and values v for every band   run once per call
  chunk.onnx   frames of the padded call + enroll_n, k, v + LSTM state -> windowed output frames + new state   run per chunk

Everything that depends only on the voice sample is computed once in enroll.onnx, so a chunk pays only for the call.

Only the six time-direction LSTMs carry anything across frames (every norm, band LSTM, attention and mask works on one
frame at a time), so chunked streaming with the LSTM state carried over equals the full-file forward pass.

Caller side (JS or Python, see stream_reference below): reflect-pad the call by 256 samples at both ends (torch.stft
center=True), feed hops of K frames (K*128 + 384 samples, consecutive chunks overlap by 384), overlap-add the returned
frames at a 128-sample hop, divide by the summed squared Hann window, drop the first 256 samples, cut to the call length.

    python export_onnx.py --ckpt ckpt --out onnx/ && python quantize.py onnx/
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
    """Everything that depends only on the voice sample, computed once per call: the normalised sample spectrum for
    the TF-map attention, and each band's attention keys and values (CrossFuse's per-band Linear, then the shared
    MultiheadAttention's key and value projections, split into heads)."""
    def __init__(self, tse):
        super().__init__()
        self.stft, self.fbank = ConvSTFT(), KaldiFbank()
        self.spk = tse.spk_ft.context.encoder.spk_model
        af = tse.spk_ft.context.attenFuse
        mha = af.multihead_attn
        d, self.heads = mha.embed_dim, mha.num_heads
        self.register_buffer("lw", torch.stack([l.weight for l in af.Linear]))         # (nband, d, 512)
        self.register_buffer("lb", torch.stack([l.bias for l in af.Linear])[:, None])  # (nband, 1, d)
        self.register_buffer("wkv", mha.in_proj_weight[d:])          # (2d, d): key and value projections
        self.register_buffer("bkv", mha.in_proj_bias[d:])

    def forward(self, enroll):
        x = F.pad(enroll[:, None], (256, 256), mode="reflect")[:, 0]
        re, im = self.stft(x)
        mag = torch.sqrt(re ** 2 + im ** 2)                          # (1, 257, Fe)
        enroll_n = F.normalize(mag, p=2, dim=1)[0]                   # (257, Fe)
        emb = self.spk._get_frame_level_feat(self.fbank(enroll))
        if isinstance(emb, tuple):
            emb = emb[-1]
        e = emb[0].T                                                 # (Fc, 512)
        nb, d = self.lw.shape[:2]
        kv = (torch.matmul(e, self.lw.transpose(1, 2)) + self.lb) @ self.wkv.T + self.bkv   # (nband, Fc, 2d)
        k, v = kv[..., :d], kv[..., d:]
        k = k.view(nb, -1, self.heads, d // self.heads).permute(0, 2, 3, 1)   # (nband, heads, d_h, Fc)
        v = v.view(nb, -1, self.heads, d // self.heads).transpose(1, 2)       # (nband, heads, Fc, d_h)
        return enroll_n, k, v

class Chunk(nn.Module):
    def __init__(self, tse):
        super().__init__()
        self.t = tse
        self.stft, self.istft = ConvSTFT(), FrameISTFT()
        sep = tse.sep_model
        self.bw = sep.band_split.band_width
        self.nband = sep.nband
        ctx = tse.spk_ft.context
        mha = ctx.attenFuse.multihead_attn
        d, self.heads = mha.embed_dim, mha.num_heads
        self.register_buffer("wq", mha.in_proj_weight[:d] / math.sqrt(d // self.heads))
        self.register_buffer("bq", mha.in_proj_bias[:d] / math.sqrt(d // self.heads))
        fc = ctx.fusionLayer.fc.linear                               # fused: x * fc(out_proj(attention))
        self.register_buffer("wo", fc.weight @ mha.out_proj.weight)
        self.register_buffer("bo", fc.weight @ mha.out_proj.bias + fc.bias)
        # Bands of equal width share one batched op: (first band, number of bands, width, first frequency bin).
        self.groups, i, f = [], 0, 0
        while i < len(self.bw):
            n = 1
            while i + n < len(self.bw) and self.bw[i + n] == self.bw[i]:
                n += 1
            self.groups.append((i, n, self.bw[i], f)); i += n; f += n * self.bw[i - n]
        bn = sep.subband_norm.BN
        for g, (i, n, bw, f) in enumerate(self.groups):
            self.register_buffer(f"in_g{g}", torch.stack([bn[j][0].weight for j in range(i, i + n)])[:, :, None])
            self.register_buffer(f"in_b{g}", torch.stack([bn[j][0].bias for j in range(i, i + n)])[:, :, None])
            self.register_buffer(f"in_w{g}", torch.stack([bn[j][1].weight[:, :, 0] for j in range(i, i + n)]))
            self.register_buffer(f"in_c{g}", torch.stack([bn[j][1].bias for j in range(i, i + n)])[:, :, None])
        self.in_eps = bn[0][0].eps
        mk = sep.band_masker.mask
        self.register_buffer("mk_g", torch.stack([m[0].weight for m in mk])[:, :, None])
        self.register_buffer("mk_b", torch.stack([m[0].bias for m in mk])[:, :, None])
        self.mk_eps = mk[0][0].eps
        for L in (1, 3):
            self.register_buffer(f"mk_w{L}", torch.stack([m[L].weight[:, :, 0] for m in mk]))
            self.register_buffer(f"mk_c{L}", torch.stack([m[L].bias for m in mk])[:, :, None])
        for g, (i, n, bw, f) in enumerate(self.groups):
            self.register_buffer(f"mk_w5_{g}", torch.stack([mk[j][5].weight[:, :, 0] for j in range(i, i + n)]))
            self.register_buffer(f"mk_c5_{g}", torch.stack([mk[j][5].bias for j in range(i, i + n)])[:, :, None])

    @staticmethod
    def layer_norm(x, g, b, eps):
        """ChannelWiseLayerNorm over dim 1 of (bands, C, K), one affine per band."""
        mu = x.mean(1, keepdim=True)
        var = ((x - mu) ** 2).mean(1, keepdim=True)
        return (x - mu) / torch.sqrt(var + eps) * g + b

    def tfmap(self, enroll_n, mag):
        """TFMap.compute with the sample side normalised once per call."""
        mix_n = F.normalize(mag, p=2, dim=1)[0].T                    # (K, 257)
        att = torch.softmax(mix_n @ enroll_n, -1)                    # (K, Fe)
        tf = (att @ enroll_n.T).T[None]                              # (1, 257, K)
        tf = tf / tf.norm(dim=1, keepdim=True)
        return torch.sum(mag * tf, dim=1, keepdim=True) * tf

    def fuse(self, feat, k, v):
        """CrossFuse + SpeakerFuseLayer("multiply"), every band at once, keys and values from Enroll."""
        nb, d, h = self.nband, self.wq.shape[0], self.heads
        x = feat[0].transpose(1, 2)                                  # (nband, K, d)
        q = (x @ self.wq.T + self.bq).view(nb, -1, h, d // h).transpose(1, 2)   # (nband, heads, K, d_h)
        a = torch.softmax(q @ k, -1) @ v                             # (nband, heads, K, d_h)
        a = a.transpose(1, 2).reshape(nb, -1, d)                     # (nband, K, d)
        return feat * (a @ self.wo.T + self.bo).transpose(1, 2)[None]

    def forward(self, wav, enroll_n, k, v, h, c):
        """wav (1, K*128+384); h, c (layers, nband, hidden) -> frames (1, K, 512), h, c."""
        t, sep = self.t, self.t.sep_model
        re, im = self.stft(wav)                                      # (1, 257, K)
        mag = torch.sqrt(re ** 2 + im ** 2)
        spec = torch.stack([re, im], 1)                              # (1, 2, 257, K)
        spec = torch.cat([spec, self.tfmap(enroll_n, mag)[:, None]], 1)   # (1, 3, 257, K)
        K = spec.shape[-1]
        feats = []
        for g, (i, n, bw, f) in enumerate(self.groups):              # BandSplit + per-band norm and projection
            x = spec[0, :, f: f + n * bw].reshape(3, n, bw, K).transpose(0, 1).reshape(n, 3 * bw, K)
            x = self.layer_norm(x, getattr(self, f"in_g{g}"), getattr(self, f"in_b{g}"), self.in_eps)
            feats.append(getattr(self, f"in_w{g}") @ x + getattr(self, f"in_c{g}"))
        feat = self.fuse(torch.cat(feats)[None], k, v)               # (1, nband, 128, K)
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
        y = self.layer_norm(x.view(nb, fd, K), self.mk_g, self.mk_b, self.mk_eps)   # BandMasker, every band at once
        y = torch.tanh(self.mk_w3 @ torch.tanh(self.mk_w1 @ y + self.mk_c1) + self.mk_c3)  # (nband, 512, K)
        er, ei = [], []
        for g, (i, n, bw, f) in enumerate(self.groups):
            o = (getattr(self, f"mk_w5_{g}") @ y[i: i + n] + getattr(self, f"mk_c5_{g}")).view(n, 2, 2, bw, K)
            m = o[:, 0] * torch.sigmoid(o[:, 1])                    # (n, 2, bw, K): real, imaginary mask
            sr, si = re[0, f: f + n * bw].view(n, bw, K), im[0, f: f + n * bw].view(n, bw, K)
            er.append((sr * m[:, 0] - si * m[:, 1]).reshape(n * bw, K)); ei.append((sr * m[:, 1] + si * m[:, 0]).reshape(n * bw, K))
        frames = self.istft(torch.cat(er)[None], torch.cat(ei)[None])   # (1, K, 512)
        return frames, torch.cat(hs), torch.cat(cs)

def hann_sq():
    return (torch.hann_window(N_FFT) ** 2).numpy()

def stream_reference(enroll_fn, chunk_fn, mix, enr, K=100, state=(6, 32, 256)):
    """The exact caller-side loop the browser runs, in numpy. Returns the isolated signal, same length as mix."""
    en, k_, v_ = enroll_fn(enr[None].astype(np.float32))
    x = np.pad(mix, (256, 256), mode="reflect").astype(np.float32)
    nfr = 1 + (len(x) - N_FFT) // HOP
    pad = (nfr - 1) * HOP + N_FFT - len(x)
    x = np.pad(x, (0, max(0, pad)))
    out = np.zeros((nfr - 1) * HOP + N_FFT, np.float32)
    env = np.zeros_like(out)
    w2 = hann_sq()
    h = np.zeros(state, np.float32); c = np.zeros_like(h)
    for f0 in range(0, nfr, K):
        k = min(K, nfr - f0)
        seg = x[f0 * HOP: (f0 + k - 1) * HOP + N_FFT]
        fr, h, c = chunk_fn(seg[None], en, k_, v_, h, c)
        for j in range(k):
            s = (f0 + j) * HOP
            out[s: s + N_FFT] += fr[0, j]; env[s: s + N_FFT] += w2
    y = out / np.where(env > 1e-11, env, 1.0)
    return y[256: 256 + len(mix)]

if __name__ == "__main__":
    from isolate import load
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", default="onnx")
    a = ap.parse_args()
    tse = load(a.ckpt, device="cpu").model.eval()
    tse.spk_ft.context.fbank.dither = 0.0
    Path(a.out).mkdir(exist_ok=True)
    en, ch = Enroll(tse).eval(), Chunk(tse).eval()
    nb = tse.sep_model.nband
    with torch.no_grad():
        e = torch.randn(1, 48000) * 0.1
        enroll_n, k, v = en(e)
        torch.onnx.export(en, (e,), f"{a.out}/enroll.onnx", input_names=["enroll"], output_names=["enroll_n", "k", "v"],
                          opset_version=17, dynamo=False)
        K = 50
        w = torch.randn(1, K * HOP + 384) * 0.1
        sep = tse.sep_model.separator.separation
        h = torch.zeros(len(sep), nb, sep[0].band_rnn.rnn.hidden_size); c = torch.zeros_like(h)
        torch.onnx.export(ch, (w, enroll_n, k, v, h, c), f"{a.out}/chunk.onnx", input_names=["wav", "enroll_n", "k", "v", "h", "c"],
                          output_names=["frames", "h_out", "c_out"], dynamic_axes={"wav": {1: "S"}, "frames": {1: "K"}},
                          opset_version=17, dynamo=False)
    import onnx, onnxsim
    for name in ("enroll", "chunk"):                                 # fold constants and shape arithmetic
        m, ok = onnxsim.simplify(onnx.load(f"{a.out}/{name}.onnx"))
        assert ok, name
        del m.metadata_props[:]                                       # onnxsim's own bookkeeping
        onnx.save(m, f"{a.out}/{name}.onnx")
    print("exported", nb, "bands")
