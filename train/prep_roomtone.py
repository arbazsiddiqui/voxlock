"""Real room tone for training: stretches of AMI headset audio where nobody is talking (labels == 0), >= 1 s each,
cut 0.3 s inside on both sides. Writes data/roomtone.npy (float16, 16 kHz)."""
import json, numpy as np
from pathlib import Path
d = Path("data/ami_train")
audio = np.load(d / "audio.npy", mmap_mode="r"); lab = np.load(d / "labels.npy", mmap_mode="r")
idx = json.loads((d / "index.json").read_text())
out, total = [], 0
for u in idx:
    l = np.asarray(lab[u["lab_off"]: u["lab_off"] + u["len"] // 160]) == 0
    edges = np.flatnonzero(np.diff(np.r_[0, l.astype(int), 0])); 
    for s, e in zip(edges[::2], edges[1::2]):
        if e - s < 160: continue
        a = np.asarray(audio[u["off"] + (s + 30) * 160: u["off"] + (e - 30) * 160]).astype(np.float32)
        if audio.dtype == np.int16: a /= 32768
        if np.abs(a).max() > 0.3: continue  # skip bumps and clicks
        out.append(a); total += len(a)
    if total > 16000 * 60 * 40: break
x = np.concatenate(out); print(f"room tone: {len(x)/16000/60:.1f} min from {len(out)} stretches, dtype {audio.dtype}")
np.save(d.parent / "roomtone.npy", x.astype(np.float16))
