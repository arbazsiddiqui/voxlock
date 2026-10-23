"""Voice samples of any length: the network takes exactly 3 s, so shorter samples are refused and longer ones give
their loudest 3 s. Runs the int8 model (downloaded on first use).

    python tests/test_sample_length.py      (or pytest tests/test_sample_length.py)
"""
import warnings

import numpy as np

import voxlock
from voxlock.core import SAMPLE, SR, fit_sample, loud_sample

rng = np.random.default_rng(0)


def noise(sec, level=0.1):
    return (rng.standard_normal(int(sec * SR)) * level).astype(np.float32)


def test_fit_sample():
    assert len(fit_sample(noise(3.0))) == SAMPLE
    for sec in (3.001, 3.005, 5.0, 40.0):
        assert len(fit_sample(noise(sec))) == SAMPLE
    quiet, loud = noise(10, 0.01), noise(3, 0.5)
    x = np.concatenate([quiet, loud, quiet])
    assert np.array_equal(fit_sample(x), loud_sample(x, within=30.0))
    for sec in (0.5, 2.0, 2.999):
        try:
            fit_sample(noise(sec))
        except ValueError:
            continue
        raise AssertionError(f"a {sec} s sample was accepted")


def test_model_takes_any_long_sample():
    vl = voxlock.VoxLock()
    call = noise(6.0)
    for sec in (3.0, 3.01, 5.0, 31.0):
        y = vl.isolate(call, sample=noise(sec))
        assert len(y) == len(call) and np.isfinite(y).all()


def test_short_call_comes_back_unchanged():
    vl = voxlock.VoxLock()
    call = noise(2.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        y = vl.isolate(call)
    assert np.array_equal(y, call) and w


def test_live_needs_three_seconds():
    vl = voxlock.VoxLock()
    vl.live(sec=4.0)
    try:
        vl.live(sec=2.0)
    except ValueError:
        return
    raise AssertionError("live(sec=2) was accepted")


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
