"""Sanas Voice Isolation SE 1.0 (model key VI_G_SE) on any set of mix.wav files, 20 ms frames, 16 kHz mono.
Output delay is measured per file by cross-correlation with the input and removed. Writes out_sanas_vi.wav.

    python sanas_run.py --glob 'krisp16/work/*/mix.wav' --procs 8 [--pace]
"""
import argparse, array, glob, os, threading, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, soundfile as sf

SR = 16000
FRAME = SR // 50

def run_one(args):
    f, pace, model, out_name = args
    import sanas
    out_p = Path(f).parent / out_name
    if out_p.exists():
        return f, "skip", 0
    sdk = sanas.create_sdk(sanas.InitParams(storage_dir=f"/tmp/sanas_{os.getpid()}"))
    for attempt in range(8):
        res = sdk.activate_api_key(os.environ["SANAS_API_KEY"])
        if res.success:
            break
        time.sleep(2 + 3 * attempt + (os.getpid() % 5))
    else:
        return f, f"activation failed {res.error_type}", 0
    x, sr = sf.read(f, dtype="float32"); assert sr == SR
    cond, state = threading.Condition(), {"v": None}
    def on_state(s):
        with cond:
            state["v"] = s; cond.notify_all()
    attrs = sanas.ProcessorAttributes(audio_attributes=sanas.AudioAttributes(
        sampling_rate=SR, channels=1, model_name=model, audio_pipeline_state_notify=on_state))
    out = []
    term = (sanas.PipelineState.RUNNING, sanas.PipelineState.NOT_RUNNING)
    with sdk.create_audio_processor(attrs) as proc:
        with cond:
            cond.wait_for(lambda: state["v"] in term, timeout=60)
        if state["v"] != sanas.PipelineState.RUNNING:
            return f, f"pipeline state {state['v']}", 0
        pad = np.concatenate([x, np.zeros(FRAME * 25, np.float32)])
        n = len(pad) // FRAME * FRAME
        t = time.monotonic()
        for i in range(0, n, FRAME):
            fr = proc.process_frame(sanas.AudioFrame(samples=array.array("f", pad[i: i + FRAME].tobytes())))
            if fr.frame_count:
                out.append(np.frombuffer(bytes(fr.samples), np.float32))
            if pace:
                t += 0.02; time.sleep(max(0.0, t - time.monotonic()))
    y = np.concatenate(out) if out else np.zeros(0, np.float32)
    print(f"{Path(f).parent.name} model={model} pace={pace} frames_out={len(out)} samples={len(y)} rms={float(np.sqrt((y**2).mean())) if len(y) else 0:.5f} max={float(np.abs(y).max()) if len(y) else 0:.4f}", flush=True)
    W = min(len(x), 10 * SR)
    maxlag = min(16000, max(len(y) - W, 0))
    if maxlag > 0 and W > 0:
        cc = np.correlate(y[: W + maxlag], x[:W], mode="valid")
        best = int(np.argmax(cc))
    else:
        best = 0
    y = y[best: best + len(x)]
    y = np.pad(y, (0, len(x) - len(y)))
    sf.write(out_p, np.clip(y, -1, 1), SR)
    return f, "ok", best

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True); ap.add_argument("--procs", type=int, default=8); ap.add_argument("--pace", action="store_true")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--model", default="VI_G_SE"); ap.add_argument("--out", default="out_sanas_vi.wav")
    a = ap.parse_args()
    files = sorted(glob.glob(a.glob))
    if a.limit:
        files = files[: a.limit]
    t0 = time.time()
    with Pool(a.procs, maxtasksperchild=1) as pool:
        for k, (f, st, lag) in enumerate(pool.imap_unordered(run_one, [(f, a.pace, a.model, a.out) for f in files]), 1):
            if st not in ("ok", "skip") or k % 10 == 0 or k == len(files):
                print(f"[{time.time()-t0:5.0f}s] sanas {k}/{len(files)} {st} lag={lag} {Path(f).parent.name}", flush=True)
    print("DONE sanas")
