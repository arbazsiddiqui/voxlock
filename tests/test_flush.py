"""A push() after flush() starts a new segment with the same voice and LSTM memory: same length out as in, and the
same audio as a new stream that starts from that memory. Runs the fp32 model (downloaded on first use).

    python tests/test_flush.py      (or pytest tests/test_flush.py)
"""
import numpy as np

import voxlock
from voxlock.core import SR

rng = np.random.default_rng(0)


def noise(sec, level=0.1):
    return (rng.standard_normal(int(sec * SR)) * level).astype(np.float32)


def run(s, x, chunk):
    return np.concatenate([s.push(x[i: i + chunk]) for i in range(0, len(x), chunk)] + [s.flush()])


def test_push_after_flush_starts_a_new_segment():
    vl = voxlock.VoxLock(precision="fp32", threads=1)
    sample, a, b = noise(4.0, 0.3), noise(2.0), noise(2.0)
    for chunk in (len(a), 640, 100):
        s = vl.stream(sample)
        assert len(run(s, a, chunk)) == len(a)
        h, c = s.h.copy(), s.c.copy()
        out_b = run(s, b, chunk)
        ref = vl.stream(sample); ref.h, ref.c = h, c
        assert len(out_b) == len(b)
        assert np.allclose(out_b, run(ref, b, chunk), atol=1e-6)


def test_flush_twice_is_empty():
    vl = voxlock.VoxLock(precision="fp32", threads=1)
    s = vl.stream(noise(4.0, 0.3))
    s.push(noise(1.0)); s.flush()
    assert len(s.flush()) == 0
    assert len(run(s, noise(1.0), 320)) == SR


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
