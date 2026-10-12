"""Fine-tune the REAL-TSE causal BSRNN to keep the caller and drop background talkers, given a dirty voice sample.

Every training example is mixed on the fly: a near-field caller (who may stop talking before the chunk ends), one or
two far-field background talkers that keep going, optional low-level noise. The enrollment is the first 3 s of the
mixture, a noisy earlier utterance of the caller, or a clean one, so the model learns to lock on from dirty samples.

    python train.py --base ckpt/tfmap_context_causal_100 --data data/train --rirs data/rirs.npz --run runs/r1
"""
import os, argparse, json, math, shutil, time
from pathlib import Path
import numpy as np, torch
from scipy.signal import butter, sosfilt, fftconvolve, resample_poly
import wesep

SR, ENR = 16000, 3 * 16000
L = int(float(os.environ.get("SEG_S", "6")) * SR)
LONG = os.environ.get("LONG_CALLER", "0") == "1"
DECOUPLE = os.environ.get("DECOUPLE", "0") == "1"  # caller distance independent of role, dry backgrounds, real room tone
ROOMTONE = os.environ.get("ROOMTONE", "")

class Mixer(torch.utils.data.IterableDataset):
    def __init__(self, data, rirs, seed):
        self.data, self.rirs, self.seed = Path(data), rirs, seed

    def _setup(self):
        self.audio = np.load(self.data / "audio.npy", mmap_mode="r")
        self.index = json.loads((self.data / "index.json").read_text())
        self.by_spk = {}
        for k, u in enumerate(self.index):
            self.by_spk.setdefault(u["spk"], []).append(k)
        self.spks = list(self.by_spk)
        r = np.load(self.rirs)
        self.near, self.far = r["near"].astype(np.float32), r["far"].astype(np.float32)
        self.mid = r["mid"].astype(np.float32) if "mid" in r else self.near
        self.tone = np.load(ROOMTONE, mmap_mode="r") if ROOMTONE else None

    def utt(self, k):
        u = self.index[k]
        return self.audio[u["off"]: u["off"] + u["len"]].astype(np.float32) / 32768

    def crop(self, x, n, rng):
        if len(x) <= n:
            return np.pad(x, (0, n - len(x)))
        s = rng.integers(0, len(x) - n)
        return x[s: s + n]

    def talker(self, spk, n, rng):
        out, filled = np.zeros(n, np.float32), 0
        while filled < n:
            x = self.utt(rng.choice(self.by_spk[spk]))
            take = min(len(x), n - filled)
            out[filled: filled + take] = x[:take]
            filled += take
        return out

    @staticmethod
    def rms(x):
        f = x[: len(x) // 320 * 320].reshape(-1, 320)
        e = (f ** 2).mean(1)
        return math.sqrt(e[e > e.max() * 1e-3].mean() + 1e-12) if e.max() > 0 else 1e-6

    def far_render(self, x, i, rng, dry_p=0.0, lp_p=0.4):
        """Background talker: 50% far (1.5-5 m), 50% as close as the caller (0.3-1.5 m); muffled only sometimes."""
        if rng.random() < dry_p:
            y = x.copy()
            if rng.random() < lp_p:
                y = sosfilt(butter(4, rng.uniform(1000, 6000), "low", fs=SR, output="sos"), y)
            return y.astype(np.float32)
        h = self.far[i] if rng.random() < 0.5 else self.mid[i]
        y = fftconvolve(x, h)[: len(x)]
        if rng.random() < lp_p:
            y = sosfilt(butter(4, rng.uniform(1000, 6000), "low", fs=SR, output="sos"), y)
        return y.astype(np.float32)

    @staticmethod
    def telephone(x):
        y = resample_poly(x, 1, 2)
        y = sosfilt(butter(4, [300, 3400], "band", fs=8000, output="sos"), y)
        mu = 255.0
        c = np.round(np.sign(y) * np.log1p(mu * np.clip(np.abs(y), 0, 1)) / np.log1p(mu) * 127) / 127
        return resample_poly(np.sign(c) * ((1 + mu) ** np.abs(c) - 1) / mu, 2, 1)[: len(x)].astype(np.float32)

    def example(self, rng):
        spk = rng.choice(self.spks)
        i = rng.integers(len(self.near))
        tail = 0 if rng.random() < 0.4 else int(rng.uniform(0.5, 3.0) * SR)
        s0 = int(rng.uniform(0, 0.5) * SR)
        cal_dry = np.zeros(L, np.float32)
        if LONG and rng.random() < 0.6:
            # the caller keeps talking: several utterances with short pauses over most of the clip (long calls)
            end, pos = L - tail, s0
            while pos < end:
                x = self.utt(rng.choice(self.by_spk[spk]))
                take = min(len(x), end - pos)
                cal_dry[pos: pos + take] = x[:take]
                pos += take + int(rng.uniform(0.15, 1.2) * SR)
        else:
            x = self.utt(rng.choice(self.by_spk[spk]))
            clen = min(len(x), L - tail - s0)
            cal_dry[s0: s0 + clen] = x[:clen]
        dec = DECOUPLE and rng.random() < 0.5  # half the examples keep the original regime as an anchor
        hc = self.near[i]
        if dec:
            u = rng.random()
            hc = self.near[i] if u < 0.5 else (self.mid[i] if u < 0.9 else self.far[i])
        cal = fftconvolve(cal_dry, hc)[:L].astype(np.float32)
        if rng.random() < (0.25 if dec else 0.2):
            cal = sosfilt(butter(4, rng.uniform(3000, 7000), "low", fs=SR, output="sos"), cal).astype(np.float32)
        bg = np.zeros(L, np.float32)
        others = [s for s in rng.choice(self.spks, 3, replace=False) if s != spk][: 1 if rng.random() < 0.8 else 2]
        for o in others:
            b = self.far_render(self.talker(o, L, rng), i, rng, dry_p=0.3 if dec else 0.0, lp_p=0.25 if dec else 0.4)
            if rng.random() < 0.3:
                a = rng.integers(0, L // 2); b[:a] = 0
            bg += b
        sir = rng.uniform(-3, 0) if rng.random() < 0.1 else rng.uniform(0, 15)
        bg *= self.rms(cal) / (self.rms(bg) * 10 ** (sir / 20))
        if LONG and rng.random() < 0.25:
            bg[:] = 0  # caller alone (plus noise below): nothing to remove, keep every word
        mix = cal + bg
        if dec and self.tone is not None and rng.random() < 0.6:
            st = rng.integers(0, len(self.tone) - L)
            t = self.tone[st: st + L].astype(np.float32)
            mix += t * self.rms(cal) / (self.rms(t) * 10 ** (rng.uniform(10, 25) / 20))
        elif rng.random() < 0.3:
            mix += rng.standard_normal(L).astype(np.float32) * self.rms(cal) * 10 ** (-rng.uniform(20, 40) / 20)
        r = rng.random()
        if r < 0.35:
            enr = mix[:ENR].copy()
        else:
            he = hc
            if dec and rng.random() < 0.3:  # enrollment heard through a different room/distance than the call
                j = rng.integers(len(self.near)); he = [self.near, self.mid, self.far][rng.integers(3)][j]
            e = fftconvolve(self.crop(self.utt(rng.choice(self.by_spk[spk])), ENR, rng), he)[:ENR].astype(np.float32)
            if r < 0.7:
                o = rng.choice([s for s in self.spks if s != spk])
                eb = self.far_render(self.talker(o, ENR, rng), i, rng, dry_p=0.3 if dec else 0.0)
                e = e + eb * self.rms(e) / (self.rms(eb) * 10 ** (rng.uniform(0, 15) / 20))
            enr = e
        if rng.random() < 0.25:
            mix, enr, cal = self.telephone(mix), self.telephone(enr), self.telephone(cal)
        g = rng.uniform(0.3, 0.9) / (np.abs(mix).max() + 1e-6)
        return (mix * g).astype(np.float32), (enr / (np.abs(enr).max() + 1e-6) * 0.9).astype(np.float32), (cal * g).astype(np.float32)

    def __iter__(self):
        w = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (w.id if w else 0) * 7919 + int(time.time()))
        self._setup()
        while True:
            yield self.example(rng)

def si_sdr_loss(est, ref, eps=1e-8):
    ref = ref - ref.mean(-1, keepdim=True); est = est - est.mean(-1, keepdim=True)
    proj = (est * ref).sum(-1, keepdim=True) * ref / (ref.pow(2).sum(-1, keepdim=True) + eps)
    noise = est - proj
    return -(10 * torch.log10(proj.pow(2).sum(-1) / (noise.pow(2).sum(-1) + eps) + eps)).mean()

def save(model, base, out):
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"models": [model.state_dict()]}, out / "avg_model.pt.tmp")
    (out / "avg_model.pt.tmp").rename(out / "avg_model.pt")
    shutil.copy(Path(base) / "config.yaml", out / "config.yaml")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--data", required=True); ap.add_argument("--rirs", required=True)
    ap.add_argument("--run", required=True); ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--every", type=int, default=500); ap.add_argument("--workers", type=int, default=6); ap.add_argument("--accum", type=int, default=1)
    a = ap.parse_args()
    run = Path(a.run); run.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = wesep.load_model_local(a.base).model.to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))))
    dl = torch.utils.data.DataLoader(Mixer(a.data, a.rirs, 0), batch_size=a.bs, num_workers=a.workers, persistent_workers=a.workers > 0)
    t0, tw, losses = time.time(), 0.0, []
    it = iter(dl)
    for step in range(1, a.steps + 1):
        opt.zero_grad(); tot = 0.0
        for _ in range(a.accum):
            tdl = time.time(); mix, enr, tgt = (x.to(dev) for x in next(it)); tw += time.time() - tdl
            out = model(mix.unsqueeze(1), enr.unsqueeze(1))
            out = out[0] if isinstance(out, (list, tuple)) else out
            loss = si_sdr_loss(out.reshape(mix.shape[0], -1)[:, : mix.shape[1]], tgt) / a.accum
            loss.backward(); tot += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step(); losses.append(tot)
        if step % 50 == 0:
            el = time.time() - t0
            print(f"step {step}/{a.steps} loss {np.mean(losses[-50:]):.2f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{el/step:.2f}s/step data-wait {tw/step:.2f}s eta {(a.steps-step)*el/step/60:.0f}min", flush=True)
        if step % a.every == 0 or step == a.steps:
            save(model, a.base, run / f"ckpt_{step:06d}")
            print(f"saved ckpt_{step:06d}", flush=True)
    (run / "TRAIN_DONE").write_text("ok")
    print("TRAIN_DONE")

if __name__ == "__main__":
    main()
