"""Krisp Voice Isolation benchmark (HF Krisp-AI/VoiceIsolation-Benchmark-Dataset, CC-BY-NC-4.0, eval only): 265 real
recordings with a competing voice in the room, scored exactly as Krisp's published methodology (Parakeet TDT 0.6B v2,
jiwer 4.0.0, their 4 regex rules + NeMo text normalization + jiwer transform, corpus-level WER). Krisp also publishes
its VI 2.5 outputs for every file, so their models are scored by this same script.

    python krisp_bench.py prep                                   # download + resample to 16 kHz under krisp16/
    python krisp_bench.py extract --ckpt runs/r3p5/ckpt_001000 --tag r3p5k1 --conds A V R
    python krisp_bench.py score --conds raw=mix.wav krisp_default=krisp_vi_2_5_default.wav --log runs/krisp_metrics.jsonl

Extraction conditions (single input unless noted):
  A  sample = first 3 s of the file
  V  sample = first 3 s of speech found by Silero VAD (the auto-sampler a packaged model would run)
  R  V, then the sample is improved from the pass-1 output (confident 3 s appended) and the file is re-run
  L  sample = loudest 3 s of the first 15 s (the closest voice on a headset or phone)
  RL L, then improved from its own output like R
  P  sample = 3 s from a labelled primary-only segment of the same file (needs a clip; reference only)
Extra metrics besides WER: deletions/insertions/substitutions (% of reference words) and leak = words the ASR emits
inside secondary-only segments (where the correct transcript is empty), per minute of secondary-only audio.
"""
import argparse, json, os, re, sys, time
from pathlib import Path
import numpy as np, soundfile as sf

SR, HOP = 16000, 160
ROOT = Path("krisp16")
SCEN = ["work", "call_center", "phone_calls"]
VARIANTS = ["vi_2_5_default", "vi_2_5_balanced", "vi_2_5_lite", "vi_2_5_hd"]

def items():
    for s in SCEN:
        for l in open(ROOT / s / "metadata.jsonl"):
            m = json.loads(l)
            yield s, m, ROOT / s / Path(m["file_name"]).stem

def prep():
    import librosa
    from huggingface_hub import snapshot_download
    raw = Path(snapshot_download("Krisp-AI/VoiceIsolation-Benchmark-Dataset", repo_type="dataset", local_dir="/tmp/krisp_raw"))
    proc = Path(snapshot_download("Krisp-AI/VoiceIsolation-Benchmark-Dataset-Processed", repo_type="dataset", local_dir="/tmp/krisp_proc"))
    n = 0
    for s in SCEN:
        (ROOT / s).mkdir(parents=True, exist_ok=True)
        (ROOT / s / "metadata.jsonl").write_text((raw / s / "metadata.jsonl").read_text())
        for l in open(raw / s / "metadata.jsonl"):
            m = json.loads(l)
            d = ROOT / s / Path(m["file_name"]).stem
            d.mkdir(exist_ok=True)
            for src, dst in [(raw / s / m["file_name"], "mix.wav")] + [(proc / s / v / m["file_name"], f"krisp_{v}.wav") for v in VARIANTS]:
                if (d / dst).exists():
                    continue
                try:
                    x, sr = sf.read(src, dtype="float32")
                except Exception as e:
                    print(f"skip unreadable {src} ({e})", flush=True); continue
                if x.ndim > 1:
                    x = x.mean(1)
                sf.write(d / dst, librosa.resample(x, orig_sr=sr, target_sr=SR) if sr != SR else x, SR)
            n += 1
        print(f"prep {s}: {n} files so far", flush=True)

def vad_sample(mix, sec=3.0):
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps
    if not hasattr(vad_sample, "m"):
        vad_sample.m = load_silero_vad()
    ts = get_speech_timestamps(torch.from_numpy(mix), vad_sample.m, sampling_rate=SR)
    out, need = [], int(sec * SR)
    for t in ts:
        seg = mix[t["start"]: t["end"]]
        out.append(seg[: need - sum(map(len, out))])
        if sum(map(len, out)) >= need:
            break
    return np.concatenate(out) if out else mix[: need]

def loud_sample(mix, sec=3.0, within=15.0):
    """Loudest `sec` of the first `within` seconds: on a headset or phone the caller is the closest, so the loudest voice."""
    seg = mix[: int(within * SR)]
    n = len(seg) // HOP
    e = (seg[: n * HOP].reshape(n, HOP) ** 2).mean(1)
    w = int(sec * SR / HOP)
    if n <= w:
        return seg
    st = int(np.convolve(e, np.ones(w), "valid").argmax()) * HOP
    return seg[st: st + int(sec * SR)]

def confident_seconds(mix, out, sec=3.0):
    n = min(len(mix), len(out)) // HOP
    em = (mix[: n * HOP].reshape(n, HOP) ** 2).mean(1) + 1e-9
    eo = (out[: n * HOP].reshape(n, HOP) ** 2).mean(1) + 1e-9
    score = np.where(10 * np.log10(eo) > 10 * np.log10(eo.max()) - 30, 10 * np.log10(eo / em), -60.0)
    w = int(sec * SR / HOP)
    if n <= w:
        return 0
    return int(np.convolve(score, np.ones(w) / w, "valid").argmax()) * HOP

def extract(ckpt, tag, conds, shard="0/1"):
    import torch, wesep
    m = wesep.load_model_local(ckpt)
    m.set_resample_rate(SR); m.set_vad(False); m.set_output_norm(False)
    m.set_device("cuda" if torch.cuda.is_available() else "cpu")
    run_full = lambda d, enr: (sf.write(d / "_enr.wav", enr, SR), m.extract_speech(str(d / "mix.wav"), str(d / "_enr.wav"))[0].cpu().numpy())[1]

    def run_chunked(d, enr, chunk=6.0, ctx=2.0):
        """Windows of `chunk` s, each run with `ctx` s of past audio as warm-up (causal: no future audio beyond the
        model's own ~16 ms). Keeps the model inside the 6 s lengths it was trained on."""
        x, _ = sf.read(d / "mix.wav", dtype="float32")
        e = torch.from_numpy(enr.astype(np.float32))[None]
        out = np.zeros_like(x)
        c, w = int(chunk * SR), int(ctx * SR)
        for st in range(0, len(x), c):
            a0 = max(0, st - w)
            seg = x[a0: st + c]
            y = m.extract_speech_from_pcm(torch.from_numpy(seg)[None], SR, e, SR)[0].cpu().numpy()
            y = np.pad(y, (0, max(0, len(seg) - len(y))))
            out[st: st + c] = y[st - a0: st - a0 + c][: len(out[st: st + c])]
        return out

    run = lambda d, enr: (run_chunked(d, enr, 8.0, 6.0) if long_ctx else run_chunked(d, enr)) if chunked else run_full(d, enr)
    t0, k = time.time(), 0
    si, sn = map(int, shard.split("/"))
    for j, (s, meta, d) in enumerate(items()):
        if j % sn != si:
            continue
        k += 1
        mix, _ = sf.read(d / "mix.wav", dtype="float32")
        todo = [c for c in conds if not (d / f"out_{tag}_{c}.wav").exists()]
        y = {}
        for c in todo:
            long_ctx = c.endswith("D")
            chunked = c.endswith("C") or long_ctx
            c0 = c[:-1] if chunked else c
            if c0 == "A":
                enr = mix[: 3 * SR]
            elif c0 in ("V", "R"):
                enr = vad_sample(mix)
            elif c0 in ("L", "RL"):
                enr = loud_sample(mix)
            elif c0 == "P":
                prim = [g for g in meta["segments"] if g["label"] == "primary" and g["end"] - g["start"] >= 3.5]
                if not prim:
                    continue
                st = int((prim[0]["start"] + 0.25) * SR); enr = mix[st: st + 3 * SR]
            base = {"R": "V", "RL": "L", "RC": "VC", "RLC": "LC", "RLD": "LD"}.get(c)
            out = y[base] if base in y else run(d, enr)
            if c0 in ("R", "RL"):
                st = confident_seconds(mix, out)
                conf = out[st: st + 3 * SR]
                out = run(d, np.concatenate([enr, conf * (np.abs(enr).max() / (np.abs(conf).max() + 1e-6))]))
            y[c] = out
            sf.write(d / f"out_{tag}_{c}.wav", out / (np.abs(out).max() + 1e-6) * 0.9, SR)
        if k % 25 == 0:
            print(f"[{time.time()-t0:5.0f}s] extract {tag} {k}/265", flush=True)
    print(f"DONE extract {tag} {time.time()-t0:.0f}s", flush=True)

# Krisp's published normalization, stage 1 (regex) + stage 2 (NeMo WFST) + stage 3 (jiwer)
GENERAL_ID_RE = re.compile(r"\b([A-Z]{1,4})(?:-?(\d+)|[ ](\d{3,}))(?:[ ]?(\d)([A-Z]{2}))?([A-Z]{0,2})\b")
DIGIT_FIRST_RE = re.compile(r"\b(\d{4,})-?([A-Za-z])\b")
UNITS = r"(?:dollars?|pounds?|euros?|percent|miles?|days?|months?|hours?|minutes?|nights?|years?|weeks?|points?|gigabytes?|megabits?)"

def split_general_id(text):
    def repl(m):
        g = m.groups()
        parts = list(g[0]) + list(g[1] or g[2] or "")
        if g[3]:
            parts += list(g[3]) + list(g[4])
        parts += list(g[5] or "")
        return " ".join(parts)
    return GENERAL_ID_RE.sub(repl, text)

def split_digit_first(text):
    def repl(m):
        if m.group(2).lower() == "s":
            return m.group(0)
        return " ".join(list(m.group(1)) + [m.group(2)])
    return DIGIT_FIRST_RE.sub(repl, text)

def split_standalone_4digits(text):
    def repl(m):
        if 1000 <= int(m.group(1)) <= 2099:
            return m.group(0)
        return " ".join(m.group(1))
    return re.sub(rf"(?<![$£€\d,])(?<!\d)\b(\d{{4}})\b(?!\d)(?!\s+{UNITS})", repl, text)

def split_adjacent_digit_groups(text):
    return re.sub(r"\b\d{1,16}(?:\s+\d{1,16}){1,}\b", lambda m: " ".join(m.group(0).replace(" ", "")), text)

def asr_run(conds):
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict
    asr = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2", map_location="cpu").cuda().eval()
    dc = asr.cfg.decoding
    with open_dict(dc):
        dc.greedy.use_cuda_graph_decoder = False
    asr.change_decoding_strategy(dc)
    shared = load_cache()
    (ROOT / "asr").mkdir(exist_ok=True)
    for spec in conds:
        name, fn = spec.split("=", 1)
        cache_p = ROOT / "asr" / f"{name}.json"
        cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
        todo = [str(d / fn) for s, m, d in items() if str(d / fn) not in cache and str(d / fn) not in shared and (d / fn).exists()]
        for i in range(0, len(todo), 32):
            hyps = asr.transcribe(todo[i: i + 32], batch_size=8, timestamps=True, verbose=False)
            for p, h in zip(todo[i: i + 32], hyps):
                cache[p] = {"text": h.text, "words": [(w["word"], w["start"], w["end"]) for w in h.timestamp["word"]]}
            print(f"  asr {name} {min(i+32, len(todo))}/{len(todo)}", flush=True)
        cache_p.write_text(json.dumps(cache))

def load_cache():
    """Shared cache from the first runs plus one file per condition (so ASR jobs can run in parallel)."""
    c = json.loads((ROOT / "asr_cache_v2.json").read_text()) if (ROOT / "asr_cache_v2.json").exists() else {}
    for f in sorted((ROOT / "asr").glob("*.json")) if (ROOT / "asr").exists() else []:
        c.update(json.loads(f.read_text()))
    return c

_NEMO = None
def _norm_init():
    global _NEMO
    from nemo_text_processing.text_normalization.normalize import Normalizer
    _NEMO = Normalizer(input_case="cased", lang="en", cache_dir=os.environ.get("TN_CACHE", "/vol/work/tn_cache"))

def _norm_one(t):
    return t, _NEMO.normalize(split_adjacent_digit_groups(split_standalone_4digits(split_digit_first(split_general_id(t)))))

def score(conds, log):
    import jiwer
    from multiprocessing import Pool
    filler = ["um", "uh", "umm", "uhh", "hmm", "mm", "mhm", "erm"]
    NORM = jiwer.Compose([jiwer.ToLowerCase(), jiwer.ExpandCommonEnglishContractions(), jiwer.SubstituteRegexes({r"[^\w\s]": " "}),
                          jiwer.RemoveSpecificWords(filler), jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ReduceToListOfListOfWords()])
    tok = lambda t: [w for w in re.sub(r"[^\w\s]", " ", t.lower()).split() if w not in filler]
    cache = load_cache()
    all_items = list(items())
    texts = {m["full_transcript"] for s, m, d in all_items}
    for spec in conds:
        fn = spec.split("=", 1)[1]
        texts |= {cache[str(d / fn)]["text"] for s, m, d in all_items if str(d / fn) in cache}
    memo_p = ROOT / "norm_cache.json"
    memo = json.loads(memo_p.read_text()) if memo_p.exists() else {}
    todo = [t for t in texts if t not in memo]
    print(f"normalizing {len(todo)} new texts on 8 processes", flush=True)
    if todo:
        _norm_init()
        with Pool(8, initializer=_norm_init) as pool:
            for k, (t, n) in enumerate(pool.imap_unordered(_norm_one, todo, chunksize=4), 1):
                memo[t] = n
                if k % 100 == 0:
                    print(f"  normalized {k}/{len(todo)}", flush=True)
        memo_p.write_text(json.dumps(memo))
    pre = memo.__getitem__
    refn = {str(d): pre(m["full_transcript"]) for s, m, d in all_items}
    for spec in conds:
        name, fn = spec.split("=", 1)
        rows = {}
        for scope in SCEN + ["all"]:
            sel = [(s, m, d) for s, m, d in all_items if (scope == "all" or s == scope) and str(d / fn) in cache]
            if not sel:
                continue
            refs = [refn[str(d)] for s, m, d in sel]
            hyps = [pre(cache[str(d / fn)]["text"]) or "<empty>" for s, m, d in sel]
            o = jiwer.process_words(refs, hyps, reference_transform=NORM, hypothesis_transform=NORM)
            N = o.hits + o.substitutions + o.deletions
            leak, sec_min = 0, 0.0
            for s, m, d in sel:
                segs = [(g["start"] + 0.25, g["end"] - 0.25) for g in m["segments"] if g["label"] == "secondary" and g["end"] - g["start"] > 0.5]
                sec_min += sum(e - b for b, e in segs) / 60
                leak += sum(len(tok(w)) for w, ws, we in cache[str(d / fn)]["words"] if any(b <= (ws + we) / 2 <= e for b, e in segs))
            rows[scope] = {"files": len(sel), "wer": round(100 * (o.substitutions + o.deletions + o.insertions) / N, 2),
                           "del": round(100 * o.deletions / N, 2), "ins": round(100 * o.insertions / N, 2), "sub": round(100 * o.substitutions / N, 2),
                           "leak_wpm": round(leak / max(sec_min, 1e-6), 2), "leak_words": leak}
        clips = []
        for s, m, d in all_items:
            if str(d / fn) not in cache:
                continue
            o = jiwer.process_words(refn[str(d)], pre(cache[str(d / fn)]["text"]) or "<empty>", reference_transform=NORM, hypothesis_transform=NORM)
            segs = [(g["start"] + 0.25, g["end"] - 0.25) for g in m["segments"] if g["label"] == "secondary" and g["end"] - g["start"] > 0.5]
            lk = sum(len(tok(w)) for w, ws, we in cache[str(d / fn)]["words"] if any(b <= (ws + we) / 2 <= e for b, e in segs))
            clips.append(dict(i=f"{s}/{d.name}", scen=s, caller=m["speaker_id"], H=o.hits, S=o.substitutions, D=o.deletions, I=o.insertions,
                              leak=lk, bgmin=round(sum(e - b for b, e in segs) / 60, 4)))
        with open(log + ".clips.jsonl", "a") as f:
            f.write(json.dumps({"set": "krisp", "cond": name, "clips": clips}) + "\n")
        row = {"time": time.strftime("%H:%M:%S"), "cond": name, **{k: v for k, v in rows.items()}}
        with open(log, "a") as f:
            f.write(json.dumps(row) + "\n")
        print("METRIC", json.dumps(row), flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "extract", "asr", "score", "warm"])
    ap.add_argument("--ckpt"); ap.add_argument("--tag"); ap.add_argument("--conds", nargs="+", default=[])
    ap.add_argument("--log", default="runs/krisp_metrics.jsonl"); ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    if a.cmd == "prep":
        prep()
    elif a.cmd == "extract":
        extract(a.ckpt, a.tag, a.conds, a.shard)
    elif a.cmd == "warm":
        from multiprocessing import Pool
        texts = sorted({m["full_transcript"] for s, m, d in items()})
        _norm_init()
        with Pool(8, initializer=_norm_init) as pool:
            memo = dict(pool.imap_unordered(_norm_one, texts, chunksize=2))
        (ROOT / "norm_cache.json").write_text(json.dumps(memo))
        print("warm", len(memo), flush=True)
    elif a.cmd == "asr":
        asr_run(a.conds)
    else:
        score(a.conds, a.log)
