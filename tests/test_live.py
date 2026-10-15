"""VoxLock.live() voice sample selection, without the model: which audio becomes the voice sample, and when.

    python tests/test_live.py      (or pytest tests/test_live.py)
"""
import numpy as np

from voxlock.core import LiveStream, SR

rng = np.random.default_rng(0)


class Recorder:
    """Stands in for VoxLock: keeps the voice sample LiveStream locks to."""
    class _Stream:
        def push(self, x, frames_per_run=100):
            return np.zeros(0, np.float32)

        def flush(self):
            return np.zeros(0, np.float32)

    def __init__(self):
        self.sample = None

    def stream(self, sample):
        self.sample = sample
        return self._Stream()


def noise(sec, dbfs):
    x = rng.standard_normal(int(sec * SR))
    return (x / np.sqrt((x ** 2).mean()) * 10 ** (dbfs / 20)).astype(np.float32)


def talker(sec, gap_ms=15, syllable_ms=250):
    """Speech-like audio with no pauses: syllables at -25 dBFS split by gaps that fall to the -60 dBFS line noise."""
    x = noise(sec, -25)
    period, gap = syllable_ms * SR // 1000, gap_ms * SR // 1000
    for s in range(period - gap, len(x), period):
        x[s: s + gap] = 0
    return x + noise(sec, -60)


def lock(audio, chunk=320):
    """The voice sample and the input time at lock, or (None, None) if it never locks."""
    vl = Recorder()
    ls = LiveStream(vl)
    for i in range(0, len(audio), chunk):
        ls.push(audio[i: i + chunk])
        if ls.locked:
            return vl.sample, (i + chunk) / SR
    return None, None


def test_locks_on_speech_without_pauses():
    sample, t = lock(talker(9.0))
    assert sample is not None and len(sample) == 3 * SR
    assert t < 4.0, t


def test_locks_after_leading_silence():
    sample, t = lock(np.concatenate([noise(2.0, -60), talker(6.0)]))
    assert sample is not None and t < 6.0, t


def test_noise_alone_never_locks():
    for x in (noise(30, -50), noise(30, -70), np.zeros(30 * SR, np.float32)):
        assert lock(x)[0] is None


def test_digital_silence_then_line_noise_stays_out_of_the_sample():
    lead = np.concatenate([np.zeros(SR, np.float32), noise(2.0, -55)])
    audio = np.concatenate([lead, talker(6.0)])
    sample, _ = lock(audio)
    frames = audio[: len(lead)].reshape(-1, 160)
    from_lead = sum(any(np.array_equal(f, g) for g in frames) for f in sample.reshape(-1, 160))
    assert from_lead <= 10, f"{from_lead} of the sample's 300 frames are line noise"


def test_chunk_size_does_not_matter():
    audio = np.concatenate([noise(1.0, -60), talker(6.0)])
    first = lock(audio, 160)[0]
    for chunk in (1, 7, 320, 4801, len(audio)):
        assert np.array_equal(lock(audio, chunk)[0], first), chunk


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
