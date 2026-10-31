"""VoxLock for voice-agent frameworks: audio frames in, frames of the same size out, a fixed delay behind, with the
model on its own thread so the framework's event loop never waits for it. The LiveKit and Pipecat filters are thin
wrappers around `Caller`.
"""

from __future__ import annotations

import atexit
import logging
import queue
import threading
import weakref

import numpy as np

from .core import SR, VoxLock

try:
    import soxr
except ModuleNotFoundError as e:
    raise ImportError("VoxLock's real-time filters need soxr: pip install soxr") from e

logger = logging.getLogger("voxlock")

LATENCY_MS = 24     # VoxLock's algorithmic latency

_models: dict[tuple, VoxLock] = {}
_models_lock = threading.Lock()
_callers: weakref.WeakSet = weakref.WeakSet()
_holds: dict[int, int] = {}


@atexit.register
def _stop_workers() -> None:
    """Let the workers leave onnxruntime before the interpreter goes: a thread killed inside it aborts the process."""
    callers = list(_callers)
    for c in callers:
        c.close()
    for c in callers:
        c.thread.join(timeout=2)


def preload(*, model_dir: str | None = None, precision: str = "int8", threads: int = 1) -> VoxLock:
    """Load VoxLock into this process ahead of the first call, e.g. when a worker process starts. Filters built with
    the same settings share it: one copy of the model per process, however many calls it runs."""
    key = (str(model_dir) if model_dir else None, precision, threads)
    with _models_lock:
        if key not in _models:
            _models[key] = VoxLock(model_dir, precision=precision, threads=threads)
        return _models[key]


def _resamplers(rate):
    """soxr streams to 16 kHz and back, and the most samples the pair holds back at `rate`."""
    if rate == SR:
        return None, None, 0
    # "LQ" holds back half what the default does and is still exact to about 100 dB, beyond what the model hears
    pair = lambda: (soxr.ResampleStream(rate, SR, 1, dtype="float32", quality="LQ"),
                    soxr.ResampleStream(SR, rate, 1, dtype="float32", quality="LQ"))
    if rate not in _holds:
        down, up = pair()
        step, n_in, n_out, worst = rate // 100, 0, 0, 0     # soxr releases audio in blocks: take the most it holds
        for _ in range(200):
            n_in += step; n_out += len(up.resample_chunk(down.resample_chunk(np.zeros(step, np.float32))))
            worst = max(worst, n_in - n_out) if n_out else worst
        _holds[rate] = worst
    return (*pair(), _holds[rate])


class Caller:
    """One caller: a VoxLock stream on its own thread, and a delay line that turns each input frame into an output
    frame of the same size.

    The audio comes in segments: a track at one rate, from its first frame until the track changes. Positions are
    sample indices within the segment. VoxLock and the resamplers keep sample i at position i and only hold back the
    tail, so the output for position p is the isolated sample p once the worker has produced it, and the input sample
    p until then. The VoxLock stream lives across segments, so a caller who toggles their microphone keeps their
    voice lock.

    The worker runs the model on `chunk_ms` of audio at a time (None: whatever has arrived) and gets one chunk's time
    to finish it, so the default delay is two chunks (or frames, if longer) plus VoxLock's 24 ms and the resampling.
    """

    def __init__(self, *, voice_sample=None, sample_seconds=3.0, threads=1, model_dir=None, precision="int8",
                 chunk_ms=None, delay_ms=None):
        self.voice_sample, self.sample_seconds = voice_sample, sample_seconds
        self.threads, self.model_dir, self.precision = threads, model_dir, precision
        self.chunk_ms, self.delay_ms = chunk_ms, delay_ms
        self.inbox: queue.SimpleQueue = queue.SimpleQueue()
        self.lock = threading.Lock()
        self.seg = 0
        self.locked = False                  # set by the worker once VoxLock isolates
        self.iso = np.zeros(0, np.float32)   # isolated audio of the current segment, from position iso_at on
        self.iso_at = 0
        self.taken = 0                       # input samples of the current segment the model has run
        self.rate = self.delay = 0
        self.closed = False
        self.thread = threading.Thread(target=self._work, name="voxlock", daemon=True)
        self.thread.start()
        _callers.add(self)

    def close(self):
        self.closed = True
        self.inbox.put(None)

    def start_segment(self, rate, n):
        """A new track or rate, or frames the framework skipped: restart the delay line, keep the voice lock."""
        _, _, hold = _resamplers(rate)
        step = max(n, rate * self.chunk_ms // 1000 if self.chunk_ms else 0)
        ms = self.delay_ms
        self.delay = int(round(rate * ms / 1000)) if ms is not None else 2 * step + rate * LATENCY_MS // 1000 + hold
        self.rate = rate
        self.raw = np.zeros(0, np.float32); self.raw_at = 0
        self.out = -self.delay
        self.gain = 0.0
        self.fade = 1.0 / max(1, rate // 100)          # 10 ms crossfades
        self.missed = 0                                # samples sent out unprocessed after the lock
        self.seg += 1
        with self.lock:
            self.iso = np.zeros(0, np.float32); self.iso_at = 0; self.taken = 0
        self.inbox.put(("segment", self.seg, rate))

    def _work(self):
        try:
            vl = preload(model_dir=self.model_dir, precision=self.precision, threads=self.threads)
            s = self.voice_sample
            stream = vl.stream(s) if s is not None else vl.live(sec=self.sample_seconds)
        except Exception:
            logger.exception("VoxLock could not load; the audio passes through unchanged")
            return
        chunk = SR * self.chunk_ms // 1000 if self.chunk_ms else 0
        seg, down, up, skip = 0, None, None, 0
        pending, n_pending, taken = [], 0, 0                   # 16 kHz audio not yet run, and its input samples
        n_in = n_out = 0                                       # 16 kHz samples into and out of the stream
        while True:
            items = [self.inbox.get()]
            while True:                                        # take everything waiting: one model run, less overhead
                try:
                    items.append(self.inbox.get_nowait())
                except queue.Empty:
                    break
            for item in items:
                if item is None:
                    return
                if item[0] == "segment":
                    seg = item[1]
                    down, up, _ = _resamplers(item[2])
                    pending, n_pending, taken = [], 0, 0
                    skip = n_in - n_out                        # the previous segment's tail, still inside the stream
                elif item[1] == seg:
                    x = down.resample_chunk(item[2]) if down else item[2]
                    pending.append(x); n_pending += len(x); taken += len(item[2])
            if n_pending < max(chunk, 1):
                continue
            x = np.concatenate(pending); pending, n_pending = [], 0
            for i in range(0, len(x), SR):                     # a second at a time: a closed call stops soon
                if self.closed:
                    return
                try:
                    y = stream.push(x[i: i + SR])
                except Exception:
                    logger.exception("VoxLock failed; the audio passes through unchanged")
                    return
                n_in += len(x[i: i + SR]); n_out += len(y)
                cut = min(skip, len(y)); skip -= cut; y = y[cut:]
                y = up.resample_chunk(y) if up else y
                with self.lock:
                    if seg == self.seg:
                        self.iso = np.concatenate([self.iso, y.astype(np.float32, copy=False)])
                        self.locked = getattr(stream, "locked", True)
            with self.lock:
                if seg == self.seg:
                    self.taken += taken
            taken = 0

    def process(self, x):
        """Input samples in, the same number out, `delay` samples behind."""
        n = len(x)
        self.raw = np.concatenate([self.raw, x])
        if self.thread.is_alive():
            self.inbox.put(("audio", self.seg, x))
        start = self.out; self.out += n
        with self.lock:
            iso, iso_at, locked = self.iso, self.iso_at, self.locked
        y = np.zeros(n, np.float32)
        lo, hi = max(start, self.raw_at), start + n             # output positions with input behind them
        if hi > lo:
            raw = self.raw[lo - self.raw_at: hi - self.raw_at]
            have = int(np.clip(iso_at + len(iso) - lo, 0, hi - lo)) if locked else 0
            self.missed += (hi - lo - have) if locked else 0
            g = np.zeros(hi - lo, np.float32)                   # share of isolated audio: fades in, snaps off
            g[:have] = np.minimum(self.gain + self.fade * np.arange(1, have + 1, dtype=np.float32), 1.0)
            self.gain = float(g[-1])
            mixed = raw * (1 - g)
            if have:
                mixed[:have] += g[:have] * iso[lo - iso_at: lo - iso_at + have]
            y[lo - start:] = mixed
        keep = max(self.out, 0)                                 # nothing before the next output position is needed
        if keep > self.raw_at:
            self.raw = self.raw[keep - self.raw_at:]; self.raw_at = keep
        with self.lock:
            if keep > self.iso_at:
                cut = min(keep - self.iso_at, len(self.iso))
                self.iso = self.iso[cut:]; self.iso_at += cut
        return y
