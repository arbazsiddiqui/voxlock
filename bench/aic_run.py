"""ai-coustics Voice Focus on any set of mix.wav files (SDK key in AIC_SDK_LICENSE; audio is processed locally, the SDK
only authorises the key and reports usage). Writes out_aic_vf_l.wav / out_aic_vf_s.wav next to mix.wav.

    python aic_run.py --glob 'krisp16/*/*/mix.wav' --models quail-vf-2.2-l-16khz quail-vf-2.2-s-16khz
"""
import argparse, glob, os, time
from pathlib import Path
import numpy as np, soundfile as sf
import aic_sdk as aic

SR = 16000
ap = argparse.ArgumentParser()
ap.add_argument("--glob", required=True); ap.add_argument("--models", nargs="+", default=["quail-vf-2.2-l-16khz", "quail-vf-2.2-s-16khz"])
a = ap.parse_args()
key = os.environ["AIC_SDK_LICENSE"]
files = sorted(glob.glob(a.glob))
for mid in a.models:
    tag = "aic_vf_" + ("l" if "-l-" in mid else "s")
    model = aic.Model.from_file(aic.Model.download(mid, Path("/tmp/aic_models")))
    cfg = aic.ProcessorConfig.optimal(model, sample_rate=SR)
    t0, audio_s = time.time(), 0.0
    for k, f in enumerate(files, 1):
        out = Path(f).parent / f"out_{tag}.wav"
        if out.exists():
            continue
        proc = aic.Processor(model, key); proc.initialize(cfg)
        delay = proc.get_context().get_audio_delay()
        x, sr = sf.read(f, dtype="float32")
        assert sr == SR
        bs = cfg.block_size
        n = len(x) + delay
        pad = np.zeros(((n + bs - 1) // bs) * bs, np.float32); pad[: len(x)] = x
        y = np.concatenate([np.asarray(proc.process(pad[i: i + bs]), np.float32) for i in range(0, len(pad), bs)])
        proc.terminate_session()
        y = y[delay: delay + len(x)]
        sf.write(out, np.clip(y, -1, 1), SR)
        audio_s += len(x) / SR
        if k % 20 == 0 or k == len(files):
            print(f"[{time.time()-t0:5.0f}s] {tag} {k}/{len(files)} ({audio_s/60:.0f} min, delay {delay} samples)", flush=True)
print("DONE aic")
