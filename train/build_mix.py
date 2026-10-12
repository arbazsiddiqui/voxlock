"""Build the Day-1 test set: 200 caller + far-field background mixtures at a fixed SIR from LibriSpeech test-clean.

Caller: near-field (0.4-0.7 m from the mic). Background: another speaker 2.5-4 m away in the same room plus a random
1-4 kHz low-pass, present from the start of the call. Each item gets an oracle enrollment (a different clean utterance
of the caller) and a self-enrollment (the first 3 s of the mixture).

    python build_mix.py --sir 5 --n 200 --out mixes_sir5
"""
import argparse, io, json, random
from pathlib import Path
import numpy as np, soundfile as sf, pyarrow.parquet as pq, pyroomacoustics as pra
from scipy.signal import butter, sosfilt, resample_poly

SR = 16000

def load_rows(path):
    t = pq.read_table(path).to_pylist()
    rows = []
    for r in t:
        wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        assert sr == SR
        rows.append({"id": r["id"], "spk": r["speaker_id"], "text": r["text"], "wav": wav})
    return rows

def active_rms(x):
    frames = x[: len(x) // 320 * 320].reshape(-1, 320)
    e = (frames ** 2).mean(1)
    keep = e > e.max() * 1e-3
    return np.sqrt(e[keep].mean() + 1e-12)

def render(src, dist, room_dim, rt60, rng):
    e_abs, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(room_dim, fs=SR, materials=pra.Material(e_abs), max_order=min(max_order, 20))
    mic = np.array([room_dim[0] / 2, room_dim[1] / 2, 1.2])
    ang = rng.uniform(0, 2 * np.pi)
    pos = mic + np.array([dist * np.cos(ang), dist * np.sin(ang), rng.uniform(-0.2, 0.4)])
    pos = np.clip(pos, 0.3, np.array(room_dim) - 0.3)
    room.add_source(pos, signal=src)
    room.add_microphone(mic)
    room.simulate()
    return room.mic_array.signals[0][: len(src)].astype(np.float32)

def telephone(x):
    """Narrowband phone channel: 8 kHz, 300-3400 Hz band, 8-bit mu-law, back to 16 kHz."""
    y = resample_poly(x, 1, 2)
    y = sosfilt(butter(4, [300, 3400], "band", fs=8000, output="sos"), y)
    mu = 255.0
    c = np.sign(y) * np.log1p(mu * np.clip(np.abs(y), 0, 1)) / np.log1p(mu)
    c = np.round(c * 127) / 127
    y = np.sign(c) * ((1 + mu) ** np.abs(c) - 1) / mu
    return resample_poly(y, 2, 1)[: len(x)].astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sir", type=float, default=5.0)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="mixes_sir5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parquet", default="test-clean.parquet")
    ap.add_argument("--bg-dist", type=float, nargs=2, default=[2.5, 4.0])
    ap.add_argument("--lowpass", type=float, nargs=2, default=[1000, 4000], help="0 0 disables the background low-pass")
    ap.add_argument("--telephony", action="store_true", help="8 kHz G.711-style channel on mix, caller and enrollments")
    ap.add_argument("--tail", type=float, default=0.0, help="seconds of background-only audio after the caller stops")
    a = ap.parse_args()
    rng = random.Random(a.seed); nrng = np.random.default_rng(a.seed)
    rows = load_rows(a.parquet)
    by_spk = {}
    for r in rows:
        by_spk.setdefault(r["spk"], []).append(r)
    callers = [r for r in rows if 6 * SR <= len(r["wav"]) <= 16 * SR]
    rng.shuffle(callers)
    out = Path(a.out); out.mkdir(exist_ok=True)
    meta = []
    for i, c in enumerate(callers[: a.n]):
        enr = rng.choice([r for r in by_spk[c["spk"]] if r["id"] != c["id"] and len(r["wav"]) >= 4 * SR])
        tail = int(a.tail * SR)
        bg_pool = [r for r in rows if r["spk"] != c["spk"] and len(r["wav"]) >= len(c["wav"]) * 0.8 + tail]
        bg = rng.choice(bg_pool)
        room = [nrng.uniform(4, 8), nrng.uniform(3, 6), nrng.uniform(2.5, 3.5)]
        rt60 = nrng.uniform(0.3, 0.7)
        cal = render(np.pad(c["wav"], (0, tail)), nrng.uniform(0.4, 0.7), room, rt60, nrng)
        b = bg["wav"][: len(cal)]
        b = np.pad(b, (0, len(cal) - len(b)))
        b = render(b, nrng.uniform(*a.bg_dist), room, rt60, nrng)
        if a.lowpass[1] > 0:
            b = sosfilt(butter(4, nrng.uniform(*a.lowpass), "low", fs=SR, output="sos"), b).astype(np.float32)
        b *= active_rms(cal[: len(c["wav"])]) / (active_rms(b) * 10 ** (a.sir / 20))
        mix = cal + b
        peak = np.abs(mix).max()
        if peak > 0.95:
            mix, cal, b = mix * 0.95 / peak, cal * 0.95 / peak, b * 0.95 / peak
        wrong = rng.choice([r for r in rows if r["spk"] != c["spk"] and len(r["wav"]) >= 4 * SR])["wav"]
        enr_w = enr["wav"]
        if a.telephony:
            mix, cal, b, enr_w, wrong = (telephone(x) for x in (mix, cal, b, enr_w, wrong))
        d = out / f"{i:03d}"; d.mkdir(exist_ok=True)
        sf.write(d / "mix.wav", mix, SR); sf.write(d / "caller.wav", cal, SR); sf.write(d / "bg.wav", b, SR)
        sf.write(d / "enroll_oracle.wav", enr_w, SR); sf.write(d / "enroll_wrong.wav", wrong, SR); sf.write(d / "enroll_self.wav", mix[: 3 * SR], SR)
        meta.append({"i": i, "caller_id": c["id"], "caller_text": c["text"], "bg_id": bg["id"], "bg_text": bg["text"],
                     "enroll_id": enr["id"], "sir": a.sir, "rt60": round(rt60, 2), "dur": round(len(mix) / SR, 2)})
        if i % 25 == 0:
            print(f"[{i}/{a.n}] built {d}", flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("DONE", len(meta))

if __name__ == "__main__":
    main()
