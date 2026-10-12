"""Run 3: continue run 2 on simulated mixtures plus real AMI headset crosstalk.

Each optimizer step = one simulated batch (SI-SDR on the rendered caller, as run 2) + one real batch of 6 s headset
windows that start at a wearer turn. Real losses use the 10 ms label track (bit 0 wearer, bit 1 others), with guard
bands for word-timing slop:
  - in 50% of real windows another participant's headset speech is injected at 0-12 dB SIR; frames where the original
    window had no crosstalk are then an exact target (masked SI-SDR against the un-injected audio)
  - wearer-only frames: hinge on output energy falling more than 1 dB below input (keep the wearer), capped at 30 dB
  - others-only frames (real crosstalk): hinge on output energy above the window's wearer level minus 30 dB, capped
Enrollment: first 3 s of the window (50%) or another turn of the same wearer (50%).

    python train3.py --base runs/r2/ckpt_004000 --data data/train --rirs data/rirs2.npz --ami data/ami_train \
        --run runs/r3 --steps 3000 --every 500
"""
import argparse, json, math, shutil, time
from pathlib import Path
import numpy as np, torch
from scipy.ndimage import binary_dilation, binary_erosion, binary_closing
import wesep
from train import Mixer, si_sdr_loss, save, SR, L, ENR
from ami_inject import AmiInject

HOP = 160

class AmiReal(torch.utils.data.IterableDataset):
    def __init__(self, ami, seed):
        self.ami, self.seed = Path(ami), seed

    def _setup(self):
        self.audio = np.load(self.ami / "audio.npy", mmap_mode="r")
        self.lab = np.load(self.ami / "labels.npy", mmap_mode="r")
        self.index = json.loads((self.ami / "index.json").read_text())
        self.by_meeting = {}
        for k, u in enumerate(self.index):
            self.by_meeting.setdefault(u["meeting"], []).append(k)
        self.onsets = []
        for k, u in enumerate(self.index):
            lab = np.asarray(self.lab[u["lab_off"]: u["lab_off"] + u["len"] // HOP])
            w = (lab & 1) > 0
            starts = np.flatnonzero(w[1:] & ~w[:-1]) + 1
            starts = starts[starts + L // HOP < len(lab)]
            self.onsets += [(k, int(s)) for s in starts]

    def get(self, k, f0, nf):
        u = self.index[k]
        a = np.asarray(self.audio[u["off"] + f0 * HOP: u["off"] + (f0 + nf) * HOP]).astype(np.float32) / 32768
        lab = np.asarray(self.lab[u["lab_off"] + f0: u["lab_off"] + f0 + nf])
        return a, lab

    @staticmethod
    def masks(lab):
        w, o = (lab & 1) > 0, (lab & 2) > 0
        w = binary_closing(w, np.ones(30))
        wo = binary_erosion(w, np.ones(15)) & ~binary_dilation(o, np.ones(15))
        oo = binary_erosion(o, np.ones(15)) & ~binary_dilation(w, np.ones(30))
        return wo, oo, o

    def example(self, rng):
        k, s = self.onsets[rng.integers(len(self.onsets))]
        nf = L // HOP
        f0 = max(0, s - int(rng.uniform(0, 0.3) * 100))
        x, lab = self.get(k, f0, nf)
        wo, oo, o = self.masks(lab)
        clean = x.copy()
        inj = np.zeros(nf, bool)
        if rng.random() < 0.5:
            others = [j for j in self.by_meeting[self.index[k]["meeting"]] if j != k]
            j = others[rng.integers(len(others))]
            uj = self.index[j]
            nj = uj["len"] // HOP
            for _ in range(20):
                g0 = int(rng.integers(0, nj - nf))
                y, labj = self.get(j, g0, nf)
                if ((labj & 1) > 0).mean() > 0.5:
                    break
            wear_rms = math.sqrt(float((x.reshape(nf, HOP)[wo] ** 2).mean()) + 1e-8) if wo.any() else math.sqrt(float((x ** 2).mean()) + 1e-8)
            y_rms = math.sqrt(float((y ** 2).mean()) + 1e-8)
            x = x + y * wear_rms / (y_rms * 10 ** (rng.uniform(0, 12) / 20))
            inj[:] = True
        if rng.random() < 0.5:
            enr = x[:ENR].copy()
        else:
            u = self.index[k]
            full = np.asarray(self.lab[u["lab_off"]: u["lab_off"] + u["len"] // HOP])
            turns = np.flatnonzero(((full & 1) > 0)[1:] & ~((full & 1) > 0)[:-1]) + 1
            far = turns[(np.abs(turns - s) > 3000) & (turns + ENR // HOP < len(full))]
            t = int(far[rng.integers(len(far))]) if len(far) else s
            enr, _ = self.get(k, t, ENR // HOP)
        g = rng.uniform(0.3, 0.9) / (np.abs(x).max() + 1e-6)
        clean_frames = (~o) & inj
        return ((x * g).astype(np.float32), (enr / (np.abs(enr).max() + 1e-6) * 0.9).astype(np.float32),
                (clean * g).astype(np.float32), wo, oo, clean_frames)

    def __iter__(self):
        w = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (w.id if w else 0) * 7919 + int(time.time()))
        self._setup()
        while True:
            yield self.example(rng)

def frame_db(x):
    f = x[..., : x.shape[-1] // HOP * HOP].reshape(*x.shape[:-1], -1, HOP)
    return 10 * torch.log10(f.pow(2).mean(-1) + 1e-8)

def real_loss(out, mix, clean, wo, oo, clean_frames):
    od, md = frame_db(out), frame_db(mix)
    n = min(od.shape[-1], wo.shape[-1])
    od, md, wo, oo, cf = od[:, :n], md[:, :n], wo[:, :n], oo[:, :n], clean_frames[:, :n]
    keep = (torch.clamp(md - od - 1.0, 0, 30) * wo).sum() / wo.sum().clamp(min=1)
    wear_db = (md * wo).sum(1, keepdim=True) / wo.sum(1, keepdim=True).clamp(min=1)
    drop = (torch.clamp(od - (wear_db - 30), 0, 30) * oo).sum() / oo.sum().clamp(min=1)
    sisdr = torch.zeros((), device=out.device)
    if cf.any():
        m = cf.repeat_interleave(HOP, dim=1)
        T = m.shape[1]
        ests, refs = [], []
        for b in range(out.shape[0]):
            if m[b].sum() > SR:
                ests.append(out[b, :T][m[b]]); refs.append(clean[b, :T][m[b]])
        if ests:
            k = min(len(e) for e in ests)
            sisdr = si_sdr_loss(torch.stack([e[:k] for e in ests]), torch.stack([r[:k] for r in refs]))
    return sisdr, keep, drop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--data", required=True); ap.add_argument("--rirs", required=True)
    ap.add_argument("--ami", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--every", type=int, default=500)
    ap.add_argument("--bs", type=int, default=4); ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--real-weight", type=float, default=1.0); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--flat-lr", action="store_true", help="constant lr after warmup instead of cosine")
    ap.add_argument("--inject", action="store_true", help="real batch = own-headset crosstalk injection with exact target (no hinges)")
    ap.add_argument("--tail", action="store_true", help="with --inject: 8 s wearer-free crosstalk canvas with a pasted wearer clip")
    a = ap.parse_args()
    run = Path(a.run); run.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    done = sorted(run.glob("ckpt_*/avg_model.pt"))
    start_dir = done[-1].parent if done else Path(a.base)
    start_step = int(start_dir.name.split("_")[1]) if done else 0
    model = wesep.load_model_local(str(start_dir)).model.to(dev).train()
    print(f"start from {start_dir} at step {start_step}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    cos = lambda s: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s + start_step, a.steps) / a.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + start_step + 1) / 100) * (1.0 if a.flat_lr else cos(s)))
    sim = iter(torch.utils.data.DataLoader(Mixer(a.data, a.rirs, 0), batch_size=a.bs, num_workers=a.workers, persistent_workers=True))
    real_ds = AmiInject(a.ami, 1, tail=a.tail) if a.inject else AmiReal(a.ami, 1)
    real = iter(torch.utils.data.DataLoader(real_ds, batch_size=a.bs, num_workers=a.workers, persistent_workers=True))
    t0, hist = time.time(), []
    for step in range(start_step + 1, a.steps + 1):
        opt.zero_grad()
        mix, enr, tgt = (x.to(dev) for x in next(sim))
        out = model(mix.unsqueeze(1), enr.unsqueeze(1)); out = out[0] if isinstance(out, (list, tuple)) else out
        ls = si_sdr_loss(out.reshape(mix.shape[0], -1)[:, : mix.shape[1]], tgt)
        ls.backward()
        if step == start_step + 1:
            gs = math.sqrt(sum(p.grad.pow(2).sum().item() for p in model.parameters() if p.grad is not None))
        if a.inject:
            rm, re_, rc = (x.to(dev) for x in next(real))
            ro = model(rm.unsqueeze(1), re_.unsqueeze(1)); ro = ro[0] if isinstance(ro, (list, tuple)) else ro
            ro = ro.reshape(rm.shape[0], -1)[:, : rm.shape[1]]
            sis = si_sdr_loss(ro, rc)
            with torch.no_grad():
                base = si_sdr_loss(rm, rc)
            keep, drop = base, torch.zeros(())
            lr_ = a.real_weight * sis
        else:
            rm, re_, rc, wo, oo, cf = next(real)
            rm, re_, rc, wo, oo, cf = rm.to(dev), re_.to(dev), rc.to(dev), wo.to(dev).float(), oo.to(dev).float(), cf.to(dev)
            ro = model(rm.unsqueeze(1), re_.unsqueeze(1)); ro = ro[0] if isinstance(ro, (list, tuple)) else ro
            ro = ro.reshape(rm.shape[0], -1)[:, : rm.shape[1]]
            sis, keep, drop = real_loss(ro, rm, rc, wo, oo, cf)
            lr_ = a.real_weight * (sis + (keep * 1.0 + drop * 0.5) / 10)
        if step == start_step + 1:
            before = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
        lr_.backward()
        if step == start_step + 1:
            gr = math.sqrt(sum((p.grad - b).pow(2).sum().item() for p, b in zip(model.parameters(), before) if p.grad is not None and b is not None))
            print(f"grad norms at first step: sim {gs:.3f} real {gr:.3f} (ratio {gr/max(gs,1e-9):.2f})", flush=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
        hist.append((ls.item(), sis.item(), keep.item(), drop.item()))
        if step % 50 == 0:
            h = np.mean(hist[-50:], 0); el = time.time() - t0; done_n = step - start_step
            print(f"step {step}/{a.steps} sim {h[0]:.2f} real_sisdr {h[1]:.2f} keep_db {h[2]:.2f} drop_db {h[3]:.2f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {el/done_n:.2f}s/step eta {(a.steps-step)*el/done_n/60:.0f}min", flush=True)
        if step % a.every == 0 or step == a.steps:
            save(model, a.base, run / f"ckpt_{step:06d}")
            print(f"saved ckpt_{step:06d}", flush=True)
    (run / "TRAIN_DONE").write_text("ok")
    print("TRAIN_DONE", flush=True)

if __name__ == "__main__":
    main()
