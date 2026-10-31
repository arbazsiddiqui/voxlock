"""VoxLockFilter inside a Pipecat pipeline: 20 ms input frames in real time at 16, 8 and 48 kHz, compared with
VoxLock's live() run offline.

    python tests/test_pipecat.py [mix.wav]      (default: the demo recording)
"""
import asyncio, sys, time
from pathlib import Path
import numpy as np, soxr

from pipecat.frames.frames import InputAudioRawFrame, StartFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_transport import TransportParams

import voxlock
from voxlock.pipecat import VoxLockFilter

MIX = Path(__file__).resolve().parent.parent / "demo" / "audio" / "bystander_000_off.mp3"


class ReadyInputTransport(BaseInputTransport):
    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)


class TimedFilter(VoxLockFilter):
    """The filter, timing each call: the transport's audio loop waits for it."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.spent = []

    async def filter(self, audio):
        t = time.perf_counter()
        out = await super().filter(audio)
        self.spent.append(time.perf_counter() - t)
        self.delay = self.delay_ms
        assert len(out) == len(audio)
        return out


async def through_pipecat(x16, rate):
    x = soxr.resample(x16, 16000, rate) if rate != 16000 else x16
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    n = rate // 50
    frames = []
    for i in range(0, len(pcm) - n + 1, n):                      # a frame every 20 ms, as from a call
        frames += [InputAudioRawFrame(audio=pcm[i: i + n].tobytes(), sample_rate=rate, num_channels=1),
                   SleepFrame(0.02)]
    f = TimedFilter()
    transport = ReadyInputTransport(TransportParams(audio_in_enabled=True, audio_in_passthrough=True,
                                                    audio_in_filter=f))
    down, _ = await run_test(Pipeline([transport]), frames_to_send=[*frames, SleepFrame(1.0)],
                             expected_down_frames=None, start_timeout=60,
                             pipeline_params=PipelineParams(audio_in_sample_rate=rate))
    out = b"".join(fr.audio for fr in down if isinstance(fr, InputAudioRawFrame))
    y = np.frombuffer(out, np.int16).astype(np.float32) / 32768
    return (soxr.resample(y, rate, 16000) if rate != 16000 else y), round(16 * f.delay), np.array(f.spent)


async def main():
    mix = voxlock.load_audio(sys.argv[1] if len(sys.argv) > 1 else MIX)
    q = (np.clip(mix, -1, 1) * 32767).astype(np.int16).astype(np.float32) / 32768   # the transport carries 16-bit
    live = voxlock.VoxLock(threads=1).live()
    ref = np.concatenate([live.push(q[i: i + 1280]) for i in range(0, len(q), 1280)])   # 80 ms chunks, as the filter
    late = slice(len(q) - 4 * 16000, len(q) - 16000)             # only the other voice, after the lock and catch-up
    below = lambda y: 10 * np.log10((q[late] @ q[late]) / (y @ y + 1e-12))
    print(f"offline live(): the other voice {below(ref[late]):.1f} dB down")
    for rate in (16000, 8000, 48000):
        y, d, spent = await through_pipecat(mix, rate)
        print(f"{rate} Hz: {len(y) / 16000:.2f} s out of {len(mix) / 16000:.2f} s in, delay {d / 16:.0f} ms, "
              f"filter() median {1000 * np.median(spent):.2f} ms, max {1000 * spent.max():.2f} ms, "
              f"the other voice {below(y[late.start + d: late.stop + d]):.1f} dB down", flush=True)
        assert spent.max() < 0.05                                # the transport never waits for the model


if __name__ == "__main__":
    asyncio.run(main())
