"""Publishable tables from per-clip rows: Krisp's real benchmark (265 recordings) and AMI real meetings (134 windows).
Pooled WER, caller words missed, deletions, insertions, leaked words per minute of background-only speech, and a
speaker-clustered bootstrap 95% interval of (model minus VoxLock) for WER, miss and leak; "*" = VoxLock better, interval
excludes zero; "!" = VoxLock worse, interval excludes zero.

    python publish_table.py --clips results/krisp.clips.jsonl --model voxlock > results/TABLE_krisp.md
"""
import argparse, json
from collections import defaultdict
import numpy as np

META = {  # name, access, real-time, input
    "voxlock": ("**VoxLock**", "open", "yes", "single"),
    "raw": ("No processing", "-", "-", "-"),
    "krisp_vi_2_5_default": ("Krisp VI 2.5 Default", "paid", "yes", "single"),
    "krisp_vi_2_5_balanced": ("Krisp VI 2.5 Balanced", "paid", "yes", "single"),
    "krisp_vi_2_5_lite": ("Krisp VI 2.5 Lite", "paid", "yes", "single"),
    "krisp_vi_2_5_hd": ("Krisp VI 2.5 HD", "paid", "yes", "single"),
    "aic_vf_l": ("ai-coustics Quail Voice Focus 2.2 L", "paid", "yes", "single"),
    "aic_vf_s": ("ai-coustics Quail Voice Focus 2.2 S", "paid", "yes", "single"),
    "sanas": ("Sanas Voice Isolation SE 1.0", "paid", "yes", "single"),
    "hush": ("Hush (weya-ai)", "open", "yes", "single"),
    "dfn3": ("DeepFilterNet3 (noise only)", "open", "yes", "single"),
    "tiger": ("TIGER + caller pick", "open", "no", "single"),
    "samaudio_s": ("SAM-Audio small (Meta), text prompt + 3 s anchor", "open", "no", "single"),
    "rtse_L": ("REAL-TSE causal, VoxLock's starting checkpoint (same auto sample)", "open", "yes", "single"),
    "rtseoff_L": ("REAL-TSE offline (same auto sample)", "open", "no", "single"),
}

def pooled(rows):
    t = {k: sum(r[k] for r in rows) for k in ("H", "S", "D", "I", "leak", "bgmin")}
    N = t["H"] + t["S"] + t["D"]
    return {"wer": 100 * (t["S"] + t["D"] + t["I"]) / N, "miss": 100 * (t["S"] + t["D"]) / N, "del": 100 * t["D"] / N,
            "ins": 100 * t["I"] / N, "leak": t["leak"] / t["bgmin"] if t["bgmin"] > 0 else float("nan")}

ap = argparse.ArgumentParser()
ap.add_argument("--clips", nargs="+", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--B", type=int, default=2000); ap.add_argument("--drop", nargs="*", default=["sanas_vi"])
a = ap.parse_args()
data = defaultdict(dict)
for f in a.clips:
    for l in open(f):
        r = json.loads(l)
        if r["cond"] in a.drop:
            continue
        data[r["set"]][r["cond"]] = {c["i"]: c for c in r["clips"]}
rng = np.random.default_rng(0)
K = ("H", "S", "D", "I", "leak", "bgmin")

def metrics(tot):
    H, S, D, I, L, B = tot
    N = H + S + D
    return {"wer": 100 * (S + D + I) / N, "miss": 100 * (S + D) / N, "del": 100 * D / N, "ins": 100 * I / N,
            "leak": L / B if B > 0 else float("nan")}

for s, conds in data.items():
    if a.model not in conds:
        continue
    ours = conds[a.model]
    title = {"krisp": "Krisp Voice Isolation Benchmark (265 real recordings, second voice in the room; Parakeet TDT 0.6B v2, Krisp's scoring)",
             "ami_headset2": "AMI real meetings, headset wearer = caller (134 windows, never used for training)"}.get(s, s)
    print(f"\n### {title}\n")
    print("| Model | Access | Real-time | WER % | Caller words missed % | Deleted % | Inserted % | Leaked words / min of background speech | WER vs VoxLock | Missed vs VoxLock | Leak vs VoxLock |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for c, cl in conds.items():
        ids = [i for i in ours if i in cl]
        if len(ids) < 0.6 * len(ours):
            continue
        X = np.array([[cl[i][k] for k in K] for i in ids], float)
        O = np.array([[ours[i][k] for k in K] for i in ids], float)
        p = metrics(X.sum(0))
        cells = ["-", "-", "-"]
        if c != a.model:
            spk = [ours[i]["caller"] for i in ids]
            uniq = sorted(set(spk)); gidx = [np.array([j for j, g in enumerate(spk) if g == u]) for u in uniq]
            diffs = {m: [] for m in ("wer", "miss", "leak")}
            for _ in range(a.B):
                pick = np.concatenate([gidx[k] for k in rng.integers(0, len(uniq), len(uniq))])
                mx, mo = metrics(X[pick].sum(0)), metrics(O[pick].sum(0))
                for m in diffs:
                    diffs[m].append(mx[m] - mo[m])
            cells = []
            for m in ("wer", "miss", "leak"):
                d = np.array(diffs[m]); lo, hi = np.nanpercentile(d, [2.5, 97.5])
                mark = " *" if lo > 0 else (" !" if hi < 0 else "")
                cells.append(f"{np.nanmean(d):+.2f} [{lo:+.2f}, {hi:+.2f}]{mark}")
        name, acc, rt, inp = META.get(c, (c, "?", "?", "?"))
        n_note = f" ({len(ids)} files)" if len(ids) < len(ours) else ""
        rows.append((p["wer"], f"| {name}{' (needs clip)' if inp == 'clip' else ''}{n_note} | {acc} | {rt} | {p['wer']:.2f} | {p['miss']:.2f} | {p['del']:.2f} | {p['ins']:.2f} | {p['leak']:.2f} | " + " | ".join(cells) + " |"))
    for _, r in sorted(rows):
        print(r)
print("\n`*` VoxLock better with the 95% interval excluding zero; `!` VoxLock worse with the interval excluding zero; intervals are a speaker-clustered bootstrap over files.")
