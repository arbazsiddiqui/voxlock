"""Real-crosstalk training data from AMI headset channels (CC-BY-4.0).

For every training meeting and participant: the participant's headset audio (int16, one memmap) and a 10 ms label
track (bit 0 = the wearer is talking, bit 1 = someone else is talking), built from the manual word timings.
Excluded: AMI's own dev/test meetings, every meeting used by REAL-T, and the 8 meetings in our ami_headset eval set.

    python ami_train_prep.py --exclude realt_ami_meetings.txt --max-meetings 40 --out data/ami_train
"""
import argparse, json, random, urllib.request, xml.etree.ElementTree as ET, zipfile
from pathlib import Path
import numpy as np, soundfile as sf

SR, HOP = 16000, 160
MIRROR = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
ANNOT = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
AMI_DEV_TEST = {f"{m}{s}" for m in ("ES2003", "ES2004", "ES2011", "IS1008", "IS1009", "TS3003", "TS3004", "EN2002") for s in "abcd"} | {
    "IB4001", "IB4002", "IB4003", "IB4004", "IB4010", "IB4011"}
OUR_EVAL = {"ES2005a", "ES2005b", "ES2006a", "ES2006b", "IS1003b", "IS1003c", "TS3005a", "TS3005b"}

def channels(root, meeting):
    for m in root.iter():
        if m.get("observation") == meeting:
            return {sp.get("nxt_agent"): int(sp.get("channel")) for sp in m if sp.get("nxt_agent") and sp.get("channel") is not None}
    return {}

def word_times(z, meeting, spk):
    name = f"words/{meeting}.{spk}.words.xml"
    if name not in z.namelist():
        return []
    return [(float(w.get("starttime")), float(w.get("endtime"))) for w in ET.fromstring(z.read(name)).iter()
            if w.tag.endswith("w") and w.get("starttime") and w.get("endtime") and (w.text or "").strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", required=True); ap.add_argument("--max-meetings", type=int, default=40)
    ap.add_argument("--out", required=True); ap.add_argument("--cache", default="ami_cache")
    a = ap.parse_args()
    cache, out = Path(a.cache), Path(a.out); cache.mkdir(exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    if not (cache / "annot.zip").exists():
        urllib.request.urlretrieve(ANNOT, cache / "annot.zip")
    z = zipfile.ZipFile(cache / "annot.zip")
    root = ET.fromstring(z.read("corpusResources/meetings.xml"))
    excl = set(Path(a.exclude).read_text().split()) | AMI_DEV_TEST | OUR_EVAL
    excl_series = {m[:6] for m in excl}
    allm = sorted({m.get("observation") for m in root.iter() if m.get("observation")})
    cands = [m for m in allm if m[:6] not in excl_series and m[:2] in ("ES", "IS", "TS")]
    print("excluded series:", " ".join(sorted(excl_series)), flush=True)
    random.Random(0).shuffle(cands)
    chunks, labels, index, off = [], [], [], 0
    used = 0
    gaps = []
    for m in cands:
        if used >= a.max_meetings:
            break
        ch = channels(root, m)
        if len(ch) < 3:
            continue
        times = {s: word_times(z, m, s) for s in ch}
        got = 0
        for spk, c in ch.items():
            wav = cache / f"{m}.Headset-{c}.wav"
            if not wav.exists():
                try:
                    urllib.request.urlretrieve(f"{MIRROR}/{m}/audio/{m}.Headset-{c}.wav", wav)
                except Exception as e:
                    print("skip", wav.name, e, flush=True); continue
            x, sr = sf.read(wav, dtype="int16")
            if x.ndim > 1:
                x = x[:, 0]
            if sr != SR:
                continue
            n = len(x) // HOP
            lab = np.zeros(n, np.int8)
            for s2, e2 in times[spk]:
                lab[int(s2 * 100): int(e2 * 100) + 1] |= 1
            for o, ws in times.items():
                if o != spk:
                    for s2, e2 in ws:
                        lab[int(s2 * 100): int(e2 * 100) + 1] |= 2
            fr = x[: n * HOP].astype(np.float32).reshape(n, HOP)
            db = 10 * np.log10((fr ** 2).mean(1) + 1e-3)
            if (lab == 1).sum() < 500 or (lab == 2).sum() < 500:
                continue
            gap = float(np.median(db[lab == 1]) - np.median(db[lab == 2]))
            gaps.append(gap)
            if gap < 6:
                print(f"drop {m}.{spk} (headset {c}): wearer-vs-others gap {gap:.1f} dB < 6, likely wrong channel map", flush=True)
                continue
            chunks.append(x[: n * HOP]); labels.append(lab)
            index.append({"meeting": m, "spk": spk, "off": off, "len": n * HOP, "lab_off": off // HOP})
            off += n * HOP; got += 1
            wav.unlink()
        used += got > 0
        print(f"[{used}/{a.max_meetings}] {m}: {got} headsets, total {off/SR/3600:.1f} h", flush=True)
    audio = np.lib.format.open_memmap(out / "audio.npy", mode="w+", dtype=np.int16, shape=(off,))
    lab_all = np.lib.format.open_memmap(out / "labels.npy", mode="w+", dtype=np.int8, shape=(off // HOP,))
    p = 0
    for x, lab in zip(chunks, labels):
        audio[p: p + len(x)] = x; lab_all[p // HOP: p // HOP + len(lab)] = lab; p += len(x)
    audio.flush(); lab_all.flush()
    (out / "index.json").write_text(json.dumps(index))
    stats = np.bincount(np.asarray(lab_all), minlength=4) / max(len(lab_all), 1)
    print(f"channel check: median wearer-vs-others gap {np.median(gaps):.1f} dB over {len(gaps)} headsets (want > 12)", flush=True)
    print(f"DONE ami_train {used} meetings, {len(index)} headsets, {off/SR/3600:.1f} h; frames silent/wearer/others/overlap = "
          + "/".join(f"{v:.2f}" for v in stats))

if __name__ == "__main__":
    main()
