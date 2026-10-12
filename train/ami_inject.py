"""Real-crosstalk examples with an exact target, from AMI headset channels.

Target windows: 4 s of one headset where the wearer talks (>= 30% of frames) and others barely do (< 10%). Interferer:
4 s of the SAME headset where only others talk (>= 70% others-only frames), i.e. real far-field crosstalk through the
same microphone, boosted 0-12 dB above its natural level and added to the target window. The un-injected target
window is the exact training target. Enrollment: the injected mixture's first 3 s (50%) or another turn of the wearer.
"""
import json, time
from pathlib import Path
import numpy as np, torch
from scipy.ndimage import binary_dilation

SR, HOP, WIN, ENR = 16000, 160, 4 * 16000, 3 * 16000

class AmiInject(torch.utils.data.IterableDataset):
    def __init__(self, ami, seed, tail=False):
        self.ami, self.seed, self.tail = Path(ami), seed, tail

    def _setup(self):
        self.audio = np.load(self.ami / "audio.npy", mmap_mode="r")
        self.lab = np.load(self.ami / "labels.npy", mmap_mode="r")
        self.index = json.loads((self.ami / "index.json").read_text())
        nf = WIN // HOP
        self.tgt, self.xtalk = {}, {}
        for k, u in enumerate(self.index):
            lab = np.asarray(self.lab[u["lab_off"]: u["lab_off"] + u["len"] // HOP])
            w, o = (lab & 1) > 0, (lab & 2) > 0
            oo = o & ~w
            wd = binary_dilation(w, np.ones(31))
            cw, co, coo = (np.concatenate([[0], np.cumsum(v)]) for v in (w, o, oo))
            starts = np.arange(0, len(lab) - nf, 50)
            fw = (cw[starts + nf] - cw[starts]) / nf
            fo = (co[starts + nf] - co[starts]) / nf
            foo = (coo[starts + nf] - coo[starts]) / nf
            t = starts[(fw >= 0.3) & (fo < 0.1)]
            if self.tail:
                cwd = np.concatenate([[0], np.cumsum(wd)])
                x = starts[((cwd[starts + nf] - cwd[starts]) == 0) & (fo >= 0.5)]
            else:
                x = starts[foo >= 0.7]
            if len(t) and len(x):
                self.tgt[k], self.xtalk[k] = t, x
        self.keys = list(self.tgt)
        print(f"AmiInject: {len(self.keys)} headsets, {sum(len(v) for v in self.tgt.values())} target windows, "
              f"{sum(len(v) for v in self.xtalk.values())} crosstalk windows", flush=True)

    def seg(self, k, f0, n):
        u = self.index[k]
        return np.asarray(self.audio[u["off"] + f0 * HOP: u["off"] + f0 * HOP + n]).astype(np.float32) / 32768

    def tail_example(self, rng):
        """8 s of wearer-free real crosstalk (natural level 40%, else +0-12 dB) with a 3-4 s wearer clip pasted in.
        Target = the wearer clip inside the paste and silence elsewhere, so every leaked sample of the tail is charged."""
        k = self.keys[rng.integers(len(self.keys))]
        xs = self.xtalk[k]
        canvas = np.concatenate([self.seg(k, int(xs[rng.integers(len(xs))]), WIN) for _ in range(2)])
        gain = 1.0 if rng.random() < 0.4 else 10 ** (rng.uniform(0, 12) / 20)
        canvas = canvas * gain
        n = int(rng.uniform(3, 4) * SR)
        clip = self.seg(k, int(self.tgt[k][rng.integers(len(self.tgt[k]))]), WIN)[:n]
        at_start = rng.random() < 0.5
        off = 0 if at_start else int(rng.integers(0, len(canvas) - n))
        tgt = np.zeros_like(canvas); tgt[off: off + n] = clip
        mix = canvas + tgt
        if at_start:
            enr = mix[:ENR].copy()
        else:
            enr = self.seg(k, int(self.tgt[k][rng.integers(len(self.tgt[k]))]), ENR)
        g = rng.uniform(0.3, 0.9) / (np.abs(mix).max() + 1e-6)
        return (mix * g).astype(np.float32), (enr / (np.abs(enr).max() + 1e-6) * 0.9).astype(np.float32), (tgt * g).astype(np.float32)

    def example(self, rng):
        if self.tail:
            return self.tail_example(rng)
        k = self.keys[rng.integers(len(self.keys))]
        t0 = int(self.tgt[k][rng.integers(len(self.tgt[k]))])
        x0 = int(self.xtalk[k][rng.integers(len(self.xtalk[k]))])
        tgt, xt = self.seg(k, t0, WIN), self.seg(k, x0, WIN)
        mix = tgt + xt * 10 ** (rng.uniform(0, 12) / 20)
        if rng.random() < 0.5:
            enr = mix[:ENR].copy()
        else:
            e0 = int(self.tgt[k][rng.integers(len(self.tgt[k]))])
            enr = self.seg(k, e0, ENR)
        g = rng.uniform(0.3, 0.9) / (np.abs(mix).max() + 1e-6)
        return (mix * g).astype(np.float32), (enr / (np.abs(enr).max() + 1e-6) * 0.9).astype(np.float32), (tgt * g).astype(np.float32)

    def __iter__(self):
        w = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (w.id if w else 0) * 7919 + int(time.time()))
        self._setup()
        while True:
            yield self.example(rng)
