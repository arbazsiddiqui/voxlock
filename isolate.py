"""Keep the caller, remove background voices. One input: the call audio. The voice sample is picked automatically.

    python isolate.py --ckpt ckpt --in call.wav --out caller.wav      # ckpt/ = config.yaml + model.safetensors
    python isolate.py --ckpt ckpt --in a.wav b.wav --out-dir out/

Voice sample = the loudest 3 s in the first 15 s of the call (on a headset or a phone the caller is the closest voice,
so the loudest one). The model is causal (about 16 ms of lookahead), so running a whole file at once gives the same
output as streaming it.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, torch

sys.path.insert(0, str(Path(__file__).parent / "wesep"))
import wesep

SR, HOP = 16000, 160

def loud_sample(mix, sec=3.0, within=15.0):
    seg = mix[: int(within * SR)]
    n = len(seg) // HOP
    e = (seg[: n * HOP].reshape(n, HOP) ** 2).mean(1)
    w = int(sec * SR / HOP)
    if n <= w:
        return seg
    st = int(np.convolve(e, np.ones(w), "valid").argmax()) * HOP
    return seg[st: st + int(sec * SR)]

def load(ckpt, device=None):
    """ckpt = a folder with config.yaml and model.safetensors (or the original avg_model.pt)."""
    st = Path(ckpt) / "model.safetensors"
    if st.exists():
        from safetensors.torch import load_model
        import wesep.cli.extractor as ext
        ext.load_pretrained_model = lambda model, path: load_model(model, str(st), strict=False)  # shared encoder weights are stored once
    m = wesep.load_model_local(ckpt)
    m.set_resample_rate(SR); m.set_vad(False); m.set_output_norm(False)
    m.set_device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return m

def read16k(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        import torchaudio
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()
    return x

def isolate(m, mix):
    enr = loud_sample(mix)
    y = m.extract_speech_from_pcm(torch.from_numpy(mix)[None], SR, torch.from_numpy(enr)[None], SR)[0].cpu().numpy()
    return np.pad(y, (0, max(0, len(mix) - len(y))))[: len(mix)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--in", dest="inp", nargs="+", required=True)
    ap.add_argument("--out"); ap.add_argument("--out-dir"); ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    m = load(a.ckpt)
    for p in a.inp:
        mix = read16k(p)
        t0 = time.time()
        y = isolate(m, mix)
        out = a.out if a.out and len(a.inp) == 1 else str(Path(a.out_dir or ".") / f"{Path(p).stem}_isolated.wav")
        sf.write(out, y, SR)
        print(f"{p} -> {out} ({len(mix) / SR:.1f} s audio in {time.time() - t0:.1f} s)", flush=True)
