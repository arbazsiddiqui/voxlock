"""Real near-field test set from AMI headset channels (CC-BY-4.0): the headset wearer is the "caller", the other
meeting participants are real crosstalk in a real room.

A window starts at one of the wearer's turns (3-15 s) and runs 4 s past it while the wearer is silent and someone else
talks, mirroring dev_tail. caller_text = the wearer's words in the window, bg_text = everyone else's (so background
word leakage is measurable). Enrollments: A = first 3 s of the window, oracle = another turn of the wearer from the
same headset, wrong = a turn of another participant. Meetings used anywhere in REAL-T are excluded.

    python ami_headset_set.py --meetings ES2005a ES2005b ... --exclude realt_ami_meetings.txt --n 200 --out ami_headset
"""
import argparse, json, random, re, urllib.request, xml.etree.ElementTree as ET, zipfile
from pathlib import Path
import numpy as np, soundfile as sf

SR = 16000
MIRROR = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
ANNOT = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"

def fetch(url, dest):
    if not Path(dest).exists():
        urllib.request.urlretrieve(url, dest)

def words(z, meeting, spk):
    name = f"words/{meeting}.{spk}.words.xml"
    if name not in z.namelist():
        return []
    out = []
    for w in ET.fromstring(z.read(name)).iter():
        if w.tag.endswith("w") and w.get("starttime") and w.get("endtime") and (w.text or "").strip():
            out.append((float(w.get("starttime")), float(w.get("endtime")), w.text.strip()))
    return out

def turns(ws, gap=0.5):
    segs = []
    for s, e, t in ws:
        if segs and s - segs[-1][1] < gap:
            segs[-1][1] = e; segs[-1][2].append(t)
        else:
            segs.append([s, e, [t]])
    return segs

def channels(z, meeting):
    root = ET.fromstring(z.read("corpusResources/meetings.xml"))
    for m in root.iter():
        if m.get("observation") == meeting:
            return {sp.get("nxt_agent"): int(sp.get("channel")) for sp in m if sp.get("nxt_agent") and sp.get("channel") is not None}
    return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", nargs="+", required=True); ap.add_argument("--exclude", required=True)
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--out", required=True); ap.add_argument("--cache", default="ami_cache")
    a = ap.parse_args()
    excl = set(Path(a.exclude).read_text().split())
    meetings = [m for m in a.meetings if m not in excl]
    print("using meetings", meetings, "excluded", [m for m in a.meetings if m in excl], flush=True)
    cache = Path(a.cache); cache.mkdir(exist_ok=True)
    fetch(ANNOT, cache / "annot.zip")
    z = zipfile.ZipFile(cache / "annot.zip")
    rng = random.Random(0)
    cands = []
    for m in meetings:
        ch = channels(z, m)
        allw = {s: words(z, m, s) for s in ch}
        for spk, c in ch.items():
            wav = cache / f"{m}.Headset-{c}.wav"
            try:
                fetch(f"{MIRROR}/{m}/audio/{m}.Headset-{c}.wav", wav)
            except Exception:
                print("missing", wav.name, flush=True); continue
            own = turns(allw[spk])
            others = [w for s, ws in allw.items() if s != spk for w in ws]
            for k, (s, e, txt) in enumerate(own):
                if not 3 <= e - s <= 15:
                    continue
                if any(s2 < e + 4 and e2 > e for s2, e2, _ in own[k + 1:k + 3]):
                    continue
                if sum(min(e2, e + 4) - max(s2, e) for s2, e2, _ in others if s2 < e + 4 and e2 > e) < 2:
                    continue
                cands.append((m, spk, c, s, e))
        print(f"{m}: {len(cands)} candidate windows so far", flush=True)
    rng.shuffle(cands)
    out = Path(a.out); out.mkdir(exist_ok=True)
    meta = []
    for i, (m, spk, c, s, e) in enumerate(cands[: a.n]):
        ch = channels(z, m)
        audio, sr = sf.read(cache / f"{m}.Headset-{c}.wav", dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        assert sr == SR
        w0, w1 = max(0.0, s - 0.2), e + 4
        mix = audio[int(w0 * SR): int(w1 * SR)]
        own = [t for t in turns(words(z, m, spk)) if t[1] - t[0] >= 3 and (t[1] < w0 - 5 or t[0] > w1 + 5)]
        if not own:
            continue
        es, ee, _ = rng.choice(own)
        oracle = audio[int(es * SR): int(min(ee, es + 8) * SR)]
        osp = rng.choice([x for x in ch if x != spk])
        ot = [t for t in turns(words(z, m, osp)) if t[1] - t[0] >= 3]
        wrong_audio, _ = sf.read(cache / f"{m}.Headset-{ch[osp]}.wav", dtype="float32")
        if wrong_audio.ndim > 1:
            wrong_audio = wrong_audio[:, 0]
        ws_, we_, _ = rng.choice(ot)
        wrong = wrong_audio[int(ws_ * SR): int(min(we_, ws_ + 8) * SR)]
        cap = " ".join(t for s2, e2, t in words(z, m, spk) if s2 >= w0 and e2 <= w1)
        bg = " ".join(t for sp in ch if sp != spk for s2, e2, t in words(z, m, sp) if s2 >= w0 and e2 <= w1)
        d = out / f"{len(meta):03d}"; d.mkdir(exist_ok=True)
        sf.write(d / "mix.wav", mix, SR); sf.write(d / "enroll_oracle.wav", oracle, SR)
        sf.write(d / "enroll_wrong.wav", wrong, SR); sf.write(d / "enroll_A.wav", mix[: 3 * SR], SR)
        meta.append({"i": len(meta), "caller_id": f"{m}-{spk}", "caller_text": re.sub(r"\s+", " ", cap),
                     "bg_text": re.sub(r"\s+", " ", bg), "meeting": m, "window": [round(w0, 2), round(w1, 2)]})
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("DONE ami headset windows", len(meta), "from", len(cands), "candidates")

if __name__ == "__main__":
    main()
