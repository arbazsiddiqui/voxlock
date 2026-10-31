"""CPU cost of one live call through the LiveKit filter: a real recording handed to VoxLockFilter as 50 ms frames in
real time, as LiveKit's RoomIO does. Reports the process CPU time (the event loop's share and the worker thread's)
per second of audio, 100% = one full core, and the time each _process call holds the event loop.

    python bench/livekit_cpu.py --audio 'krisp16/work/*.wav' --rates 16000,24000
"""
import argparse, glob, sys, time
from pathlib import Path
import numpy as np
import soxr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from livekit import rtc
from voxlock.core import SR, load_audio
from voxlock.livekit import VoxLockFilter, preload


def call_audio(pattern, sec):
    parts, n = [], 0
    for f in sorted(glob.glob(pattern)):
        a = load_audio(f)
        parts.append(a); n += len(a)
        if n >= sec * SR:
            break
    return np.concatenate(parts)[: int(sec * SR)]


def measure(audio, rate, frame_ms):
    x = soxr.resample(audio, SR, rate) if rate != SR else audio
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    n = rate * frame_ms // 1000
    f = VoxLockFilter()
    held = []
    c0, t0 = time.process_time(), time.perf_counter()
    for k, i in enumerate(range(0, len(pcm) - n + 1, n)):
        time.sleep(max(0.0, t0 + k * frame_ms / 1000 - time.perf_counter()))
        fr = rtc.AudioFrame(data=pcm[i: i + n].tobytes(), sample_rate=rate, num_channels=1, samples_per_channel=n)
        t = time.perf_counter()
        f._process(fr)
        held.append(time.perf_counter() - t)
    cpu = time.process_time() - c0
    missed = f._caller.missed / rate
    f._close()
    held = np.array(held) * 1000
    return 100 * cpu / (len(x) / rate), np.median(held), held.max(), missed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="glob of recordings, joined into one call")
    ap.add_argument("--sec", type=float, default=60)
    ap.add_argument("--rates", default="16000,24000")
    ap.add_argument("--frame-ms", type=int, default=50)
    a = ap.parse_args()
    audio = call_audio(a.audio, a.sec)
    preload()
    print(f"{len(audio) / SR:.0f} s of audio, {a.frame_ms} ms frames, int8, one thread")
    print("rate   %_of_one_core  loop_ms_median  loop_ms_max  s_unprocessed_after_lock")
    for rate in map(int, a.rates.split(",")):
        pct, med, mx, missed = measure(audio, rate, a.frame_ms)
        print(f"{rate:5d}  {pct:13.1f}  {med:14.2f}  {mx:11.2f}  {missed:.2f}", flush=True)


if __name__ == "__main__":
    main()
