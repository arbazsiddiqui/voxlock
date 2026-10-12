"""Scorer v2 (after the Fable audit): time-based leak, caller miss, per-clip rows for clustered bootstrap.

  leak     = ASR words whose midpoint falls where only the background talks (caller silent, 0.25 s guard), per minute of
             that background-only audio. Sim sets: regions from caller.wav / bg.wav energy. AMI sets: from the manual
             word timings (ami_times.json written by ami_times.py).
  miss     = (deletions + substitutions) / caller words: caller words not reproduced, whether lost or overwritten.
  wer/del/ins/sub as usual. Text normalisation = Krisp's jiwer transform (lowercase, contractions, punctuation, fillers).
Per-clip rows go to <log>.clips.jsonl so scoreboard.py can bootstrap by caller.

    python score_v2.py --mixes ami_headset2 --conds raw=mix.wav r3p5k1_A=out_r3p5k1_A.wav --log runs/v2_metrics.jsonl
"""
import argparse, json, time
from pathlib import Path
import numpy as np, soundfile as sf, jiwer, torch
from omegaconf import open_dict

SR, HOP, GUARD = 16000, 160, 0.25
FILLER = ["um", "uh", "umm", "uhh", "hmm", "mm", "mhm", "erm"]
NORM = jiwer.Compose([jiwer.ToLowerCase(), jiwer.ExpandCommonEnglishContractions(), jiwer.SubstituteRegexes({r"[^\w\s]": " "}),
                      jiwer.RemoveSpecificWords(FILLER), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])

def active(path, n):
    x, _ = sf.read(path, dtype="float32")
    x = x[: n * HOP] if len(x) >= n * HOP else np.pad(x, (0, n * HOP - len(x)))
    db = 10 * np.log10((x.reshape(n, HOP) ** 2).mean(1) + 1e-10)
    a = db > db.max() - 35
    k = int(0.2 * SR / HOP)
    return np.convolve(a, np.ones(k), "same") > 0

def intervals(mask):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [(s * HOP / SR, e * HOP / SR) for s, e in zip(edges[::2], edges[1::2])]

def bg_only_regions(d, m, times):
    if times is not None:
        t = times[str(m["i"])]
        n = int(t["dur"] * 100) + 1
        cal, oth = np.zeros(n, bool), np.zeros(n, bool)
        for s, e in t["caller"]:
            cal[max(0, int((s - GUARD) * 100)): int((e + GUARD) * 100) + 1] = True
        for s, e in t["others"]:
            oth[max(0, int(s * 100)): int(e * 100) + 1] = True
        r = oth & ~cal
        return [(s * HOP / SR, e * HOP / SR) for s, e in intervals_frames(r)]
    mix, _ = sf.read(d / "mix.wav", dtype="float32")
    n = len(mix) // HOP
    cal = active(d / "caller.wav", n)
    cal = np.convolve(cal, np.ones(int(2 * GUARD * SR / HOP)), "same") > 0
    return intervals(active(d / "bg.wav", n) & ~cal)

def intervals_frames(mask):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2]))

ap = argparse.ArgumentParser()
ap.add_argument("--mixes", required=True); ap.add_argument("--conds", nargs="+", required=True)
ap.add_argument("--log", required=True); ap.add_argument("--asr", default="nvidia/parakeet-tdt-0.6b-v3")
a = ap.parse_args()
root = Path(a.mixes)
meta = json.loads((root / "meta.json").read_text())
items = [root / f"{m['i']:03d}" for m in meta]
times = json.loads((root / "ami_times.json").read_text()) if (root / "ami_times.json").exists() else None
import nemo.collections.asr as nemo_asr
asr = nemo_asr.models.ASRModel.from_pretrained(a.asr, map_location="cpu").cuda().eval()
dc = asr.cfg.decoding
with open_dict(dc):
    dc.greedy.use_cuda_graph_decoder = False
asr.change_decoding_strategy(dc)
cache_p = root / f"asr_ts_{a.asr.split('/')[-1]}.json"
cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
regions = {str(d): bg_only_regions(d, m, times) for m, d in zip(meta, items)}
for spec in a.conds:
    name, fn = spec.split("=", 1)
    todo = [str(d / fn) for d in items if str(d / fn) not in cache and (d / fn).exists()]
    for i in range(0, len(todo), 32):
        hyps = asr.transcribe(todo[i: i + 32], batch_size=8, timestamps=True, verbose=False)
        for p, h in zip(todo[i: i + 32], hyps):
            cache[p] = {"text": h.text, "words": [(w["word"], w["start"], w["end"]) for w in h.timestamp["word"]]}
    if todo:
        cache_p.write_text(json.dumps(cache))
    clips, tot = [], dict(H=0, S=0, D=0, I=0, leak=0, bgmin=0.0)
    for m, d in zip(meta, items):
        if str(d / fn) not in cache:
            continue
        ref = NORM(m["caller_text"]) or "<none>"
        hyp = NORM(cache[str(d / fn)]["text"]) or "<empty>"
        o = jiwer.process_words(ref, hyp)
        regs = regions[str(d)]
        leak = sum(len(NORM(w).split()) for w, s, e in cache[str(d / fn)]["words"] if any(b <= (s + e) / 2 <= c for b, c in regs))
        bgmin = sum(c - b for b, c in regs) / 60
        row = dict(i=m["i"], caller=m.get("caller_id", str(m["i"])), H=o.hits, S=o.substitutions, D=o.deletions, I=o.insertions,
                   leak=leak, bgmin=round(bgmin, 4))
        clips.append(row)
        for k in ("H", "S", "D", "I", "leak", "bgmin"):
            tot[k] += row[k]
    N = tot["H"] + tot["S"] + tot["D"]
    out = {"time": time.strftime("%H:%M:%S"), "set": root.name, "cond": name, "asr": a.asr.split("/")[-1], "clips": len(clips),
           "wer": round(100 * (tot["S"] + tot["D"] + tot["I"]) / N, 2), "miss": round(100 * (tot["S"] + tot["D"]) / N, 2),
           "del": round(100 * tot["D"] / N, 2), "ins": round(100 * tot["I"] / N, 2),
           "leak_wpm": round(tot["leak"] / max(tot["bgmin"], 1e-6), 2), "bg_min": round(tot["bgmin"], 1)}
    with open(a.log, "a") as f:
        f.write(json.dumps(out) + "\n")
    with open(a.log + ".clips.jsonl", "a") as f:
        f.write(json.dumps({"set": root.name, "cond": name, "asr": out["asr"], "clips": clips}) + "\n")
    print("METRIC", json.dumps(out), flush=True)
