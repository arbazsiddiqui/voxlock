"""VoxLockFilter as LiveKit runs it: 50 ms frames handed to _process one by one, in real time, at 16, 24 and 48 kHz,
compared with VoxLock's live() run offline. Runs the int8 model (downloaded on first use).

    python tests/test_livekit.py      (or pytest tests/test_livekit.py)
"""
import time
from pathlib import Path

import numpy as np
import soxr

from livekit import rtc

import voxlock
from voxlock.livekit import VoxLockFilter

MIX = Path(__file__).resolve().parent.parent / "demo" / "audio" / "bystander_000_off.mp3"
CALLER = slice(7 * 16000, 9 * 16000 + 12800)            # the caller talks, well after the lock-in at about 3 s
OTHER = slice(10 * 16000 + 3200, 13 * 16000 + 12800)    # only the other voice


def frames(x, rate, ms=50):
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    n = rate * ms // 1000
    for i in range(0, len(pcm) - n + 1, n):
        yield rtc.AudioFrame(data=pcm[i: i + n].tobytes(), sample_rate=rate, num_channels=1, samples_per_channel=n)


def run(x16, rate, f=None, realtime=True, on_frame=None, wait=False):
    """The mix through the filter as LiveKit would feed it. Returns the output at 16 kHz, the delay in 16 kHz samples
    and the time each _process call took. With `wait`, each frame waits for the model to finish the ones before it,
    as on a CPU fast enough to keep up from the first frame: the output then does not depend on the machine."""
    x = soxr.resample(x16, voxlock.SR, rate) if rate != voxlock.SR else x16
    f = f or VoxLockFilter()
    out, spent = [], []
    t0 = time.perf_counter()
    for k, fr in enumerate(frames(x, rate)):
        if on_frame:
            on_frame(k, f)
        if realtime:                                             # frames arrive every 50 ms, as from a call
            time.sleep(max(0.0, t0 + k * 0.05 - time.perf_counter()))
        if wait and f._caller is not None:
            c = f._caller
            while c.taken < k * fr.samples_per_channel and c.thread.is_alive():
                time.sleep(0.001)
        t = time.perf_counter()
        y = f._process(fr)
        spent.append(time.perf_counter() - t)
        assert y.samples_per_channel == fr.samples_per_channel and y.sample_rate == rate
        out.append(np.frombuffer(y.data, np.int16))
    delay = round(f.delay_ms * 16)
    f._close()
    y = np.concatenate(out).astype(np.float32) / 32768
    return (soxr.resample(y, rate, voxlock.SR) if rate != voxlock.SR else y), delay, np.array(spent)


def snr(ref, y):
    e = ref - y
    return 10 * np.log10((ref @ ref) / (e @ e + 1e-12))


def below(inp, out):
    """dB the output is below the input."""
    return 10 * np.log10((inp @ inp) / (out @ out + 1e-12))


mix = voxlock.load_audio(MIX)
q = (np.clip(mix, -1, 1) * 32767).astype(np.int16).astype(np.float32) / 32768   # the track carries 16-bit audio
offline = voxlock.VoxLock(threads=1).live()
ref = np.concatenate([offline.push(q[i: i + 800]) for i in range(0, len(q), 800)])   # 50 ms chunks, as the filter


def test_same_size_frames_and_the_input_before_the_lock():
    y, d, _ = run(mix, 16000, realtime=False)
    assert len(y) == len(q) // 800 * 800
    assert np.all(y[:d] == 0)
    assert np.abs(y[d: d + 32000] - q[:32000]).max() <= 2 / 32768   # before VoxLock has 3 s of the caller: the input


def test_matches_offline_once_isolating():
    for rate in (16000, 24000, 48000):
        f = VoxLockFilter()
        y, d, _ = run(mix, rate, f=f, realtime=False, wait=True)
        s = snr(ref[CALLER], y[CALLER.start + d: CALLER.stop + d])
        print(f"{rate} Hz: delay {d / 16:.0f} ms, output vs offline live() {s:.1f} dB")
        assert s > (40 if rate == 16000 else 20)
        assert below(q[OTHER], y[OTHER.start + d: OTHER.stop + d]) > 40   # only the other voice: removed


def test_real_time():
    """Frames every 50 ms: the event loop never waits for the model, and once the model has caught up on the call
    (a few seconds after the lock, depending on the CPU) the other voice is gone."""
    y, d, spent = run(mix, 16000)
    print(f"_process median {1000 * np.median(spent):.2f} ms, max {1000 * spent.max():.2f} ms")
    assert np.median(spent) < 0.002
    late = slice(12 * 16000, 13 * 16000 + 12800)
    assert below(q[late], y[late.start + d: late.stop + d]) > 30


def test_pre_connect_audio_passes_through():
    f = VoxLockFilter()
    fr = next(frames(q, 48000, ms=100))

    def _apply_audio_processor():                                # the name RoomIO replays pre-connect audio under
        return f._process(fr)

    assert _apply_audio_processor() is fr and f._caller is None


def test_track_change_keeps_the_lock_new_participant_starts_over():
    def toggle(k, f):
        if k == 0:
            f._on_stream_info_updated(room_name="r", participant_identity="caller", publication_sid="TR_1")
        if k == 200:                                             # 10 s in: the caller's microphone comes back
            f._on_stream_info_updated(room_name="r", participant_identity="caller", publication_sid="TR_2")

    y, d, _ = run(mix, 16000, on_frame=toggle)
    assert below(q[OTHER], y[OTHER.start + d: OTHER.stop + d]) > 30

    def stranger(k, f):
        toggle(k, f) if k < 200 else None
        if k == 200:
            f._on_stream_info_updated(room_name="r", participant_identity="someone else", publication_sid="TR_3")

    y, d, _ = run(mix, 16000, on_frame=stranger)
    assert below(q[OTHER], y[OTHER.start + d: OTHER.stop + d]) < 1   # no voice sample yet: the input goes out


if __name__ == "__main__":
    test_same_size_frames_and_the_input_before_the_lock()
    test_matches_offline_once_isolating()
    test_real_time()
    test_pre_connect_audio_passes_through()
    test_track_change_keeps_the_lock_new_participant_starts_over()
    print("ok")
