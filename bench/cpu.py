"""CPU cost of one live call: stream a real recording through VoxLock in fixed chunks, as a voice agent would.

Reports the share of one CPU thread the call takes (process CPU time / audio time; 100% = one full thread) and the
real-time factor (wall time / audio time, below 1 keeps up). Includes the Python side of the streaming loop.

    python bench/cpu.py --models onnx/ --precision int8 --audio 'ami/*/mix.wav' --chunk-ms 10,20,40,80 --threads 1,2,4
"""
import argparse, glob, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voxlock.core import SR, VoxLock, load_audio, loud_sample


def call_audio(pattern, sec):
    """Real meeting audio, recordings joined until `sec` seconds."""
    parts, n = [], 0
    for f in sorted(glob.glob(pattern)):
        a = load_audio(f)
        parts.append(a); n += len(a)
        if n >= sec * SR:
            break
    return np.concatenate(parts)[: int(sec * SR)]


def measure(vl, audio, chunk_ms, repeats):
    step = SR * chunk_ms // 1000
    best = None
    for _ in range(repeats):
        s = vl.stream(loud_sample(audio))
        c0, w0 = time.process_time(), time.perf_counter()
        for i in range(0, len(audio), step):
            s.push(audio[i: i + step])
        s.flush()
        cpu, wall = time.process_time() - c0, time.perf_counter() - w0
        best = (cpu, wall) if best is None or cpu < best[0] else best
    sec = len(audio) / SR
    return 100 * best[0] / sec, best[1] / sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--precision", default="int8")
    ap.add_argument("--chunk-ms", default="10,20,40,80")
    ap.add_argument("--threads", default="1")
    ap.add_argument("--audio", required=True, help="glob of 16 kHz recordings, joined into one call")
    ap.add_argument("--sec", type=float, default=30)
    ap.add_argument("--repeats", type=int, default=2)
    a = ap.parse_args()
    audio = call_audio(a.audio, a.sec)
    print(f"{a.models} {a.precision}, {len(audio) / SR:.0f} s of audio")
    print("threads  chunk_ms  %_of_one_thread  rtf")
    for t in map(int, a.threads.split(",")):
        vl = VoxLock(a.models, precision=a.precision, threads=t)
        for ms in map(int, a.chunk_ms.split(",")):
            pct, rtf = measure(vl, audio, ms, a.repeats)
            print(f"{t:7d}  {ms:8d}  {pct:15.1f}  {rtf:.3f}", flush=True)


if __name__ == "__main__":
    main()
