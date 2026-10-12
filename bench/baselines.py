"""Open single-input baselines on any set of mix.wav files: Hush (weya-ai/hush, DeepFilterNet3 trained to keep one
speaker, Apache-2.0) and plain DeepFilterNet3 (noise suppressor, MIT). Writes out_hush.wav / out_dfn3.wav next to mix.wav.

    python baselines.py --glob 'krisp16/*/*/mix.wav' --models hush dfn3
"""
import argparse, glob, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, torch, librosa

SR = 16000
ap = argparse.ArgumentParser()
ap.add_argument("--glob", required=True); ap.add_argument("--models", nargs="+", default=["hush", "dfn3"])
ap.add_argument("--hush-dir", default="/opt/hush"); ap.add_argument("--hush-ckpt", default="/opt/hush_ckpt/model_best.ckpt")
a = ap.parse_args()
torch.set_num_threads(8)
files = sorted(glob.glob(a.glob))
run = {}
if "hush" in a.models:
    sys.path.insert(0, a.hush_dir); sys.path.insert(0, a.hush_dir + "/scripts")
    import infer_single as hs
    cfg = hs.get_config(); hm = hs.DfNetSE(cfg).eval()
    sd = torch.load(a.hush_ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    try:
        hm.model.load_state_dict(sd, strict=True)
    except RuntimeError:
        try:
            hm.load_state_dict(sd, strict=True)
        except RuntimeError:
            hm.model.load_state_dict({k[6:]: v for k, v in sd.items()}, strict=True)
    def hush(x):
        with torch.no_grad():
            return hs.enhance(hm, torch.from_numpy(x)[None], pad_delay=True, atten_lim_db=None).squeeze().numpy()
    run["hush"] = hush
if "dfn3" in a.models:
    from df.enhance import enhance as df_enhance, init_df
    dm, dstate, _ = init_df()
    def dfn3(x):
        y = librosa.resample(x, orig_sr=SR, target_sr=dstate.sr())
        with torch.no_grad():
            e = df_enhance(dm, dstate, torch.from_numpy(y)[None]).squeeze().numpy()
        return librosa.resample(e, orig_sr=dstate.sr(), target_sr=SR)[: len(x)]
    run["dfn3"] = dfn3
t0 = time.time(); audio_s = 0.0
for k, f in enumerate(files, 1):
    x, sr = sf.read(f, dtype="float32")
    assert sr == SR
    for name, fn in run.items():
        out = Path(f).parent / f"out_{name}.wav"
        if not out.exists():
            y = fn(x)
            sf.write(out, y / (np.abs(y).max() + 1e-6) * 0.9, SR)
    audio_s += len(x) / SR
    if k % 20 == 0 or k == len(files):
        print(f"[{time.time()-t0:5.0f}s] baselines {k}/{len(files)} ({audio_s/60:.0f} min audio)", flush=True)
print("DONE baselines")
