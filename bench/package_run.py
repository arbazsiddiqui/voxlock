"""Run the voxlock package over benchmark recordings and write out_<tag>.wav next to each mix.wav, for krisp_bench.py
score and score_v2.py. File mode is isolate() with its own voice sample (the loudest 3 s of the first 15 s), the
benchmarked protocol; --stream-ms runs the same sample through push() in chunks of that length, as a live agent does.

    python bench/package_run.py --models onnx/ --precision int8 --tag int8 --mixes 'krisp16/*/*/mix.wav'
    python bench/package_run.py --models onnx/ --precision int8 --tag int8_s20 --stream-ms 20 --mixes 'krisp16/*/*/mix.wav'
"""
import argparse, glob, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def init(models, precision, threads):
    global vl
    from voxlock.core import VoxLock
    vl = VoxLock(models, precision=precision, threads=threads)


def one(job):
    f, tag, stream_ms = job
    from voxlock.core import SR, loud_sample
    out = Path(f).parent / f"out_{tag}.wav"
    if out.exists():
        return
    x, _ = sf.read(f, dtype="float32")
    if not stream_ms:
        y = vl.isolate(x)
    else:
        s, step = vl.stream(loud_sample(x)), SR * stream_ms // 1000
        y = np.concatenate([s.push(x[i: i + step]) for i in range(0, len(x), step)] + [s.flush()])[: len(x)]
    sf.write(out, y, SR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True); ap.add_argument("--precision", required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--mixes", required=True)
    ap.add_argument("--stream-ms", type=int, default=0)
    ap.add_argument("--procs", type=int, default=8); ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    files = sorted(glob.glob(a.mixes))
    with Pool(a.procs, initializer=init, initargs=(a.models, a.precision, a.threads)) as p:
        for k, _ in enumerate(p.imap_unordered(one, [(f, a.tag, a.stream_ms) for f in files]), 1):
            if k % 50 == 0 or k == len(files):
                print(f"{a.tag} {k}/{len(files)}", flush=True)


if __name__ == "__main__":
    main()
