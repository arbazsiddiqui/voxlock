"""VoxLockFilter inside a Pipecat pipeline: 20 ms input frames at 8, 16 and 48 kHz, compared with VoxLock run offline.

    python tests/test_pipecat.py <mix.wav> [caller.wav]
"""
import asyncio, sys
import numpy as np, soundfile as sf, soxr

from pipecat.frames.frames import InputAudioRawFrame, StartFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_transport import TransportParams

import voxlock
from voxlock.pipecat import VoxLockFilter


class ReadyInputTransport(BaseInputTransport):
    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)


async def through_pipecat(x16, rate):
    x = soxr.resample(x16, 16000, rate) if rate != 16000 else x16
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    n = rate // 50
    frames = [InputAudioRawFrame(audio=pcm[i: i + n].tobytes(), sample_rate=rate, num_channels=1)
              for i in range(0, len(pcm) - n + 1, n)]
    transport = ReadyInputTransport(TransportParams(audio_in_enabled=True, audio_in_passthrough=True,
                                                    audio_in_filter=VoxLockFilter()))
    down, _ = await run_test(Pipeline([transport]), frames_to_send=[*frames, SleepFrame(2.0 + 1.5 * len(x16) / 16000)],
                             expected_down_frames=None, start_timeout=60,
                             pipeline_params=PipelineParams(audio_in_sample_rate=rate))
    out = b"".join(f.audio for f in down if isinstance(f, InputAudioRawFrame))
    y = np.frombuffer(out, np.int16).astype(np.float32) / 32768
    return soxr.resample(y, rate, 16000) if rate != 16000 else y


def other_below_caller(y, cal, bg, i0):
    """dB of caller over the other voice in y after sample i0, by least-squares fits of each voice to y."""
    y, cal, bg = y[i0:], cal[i0:], bg[i0:]
    lags = range(-16, 48)
    A = np.stack([np.roll(cal, k) for k in lags] + [np.roll(bg, k) for k in lags], 1)
    h, *_ = np.linalg.lstsq(A[::3], y[::3], rcond=1e-4)
    yc, yb = A[:, :len(lags)] @ h[:len(lags)], A[:, len(lags):] @ h[len(lags):]
    return 10 * np.log10((yc @ yc) / (yb @ yb))


async def main():
    mix = voxlock.load_audio(sys.argv[1])
    cal = voxlock.load_audio(sys.argv[2]) if len(sys.argv) > 2 else None
    vl = voxlock.VoxLock(threads=4)
    live = vl.live()
    q = (np.clip(mix, -1, 1) * 32767).astype(np.int16).astype(np.float32) / 32768   # the transport carries 16-bit audio
    c = VoxLockFilter()._chunk                                                       # the voice sample locks at a chunk boundary
    ref = np.concatenate([live.push(q[i: i + c]) for i in range(0, len(q), c)])
    for rate in (16000, 8000, 48000):
        y = await through_pipecat(mix, rate)
        m = min(len(y), len(ref))
        line = f"{rate} Hz: {len(y) / 16000:.2f} s out of {len(mix) / 16000:.2f} s in"
        if rate == 16000:
            line += f", max difference from offline {np.abs(y[:m] - ref[:m]).max():.2e}"
        if cal is not None:
            bg = mix - cal
            k = min(len(y), len(cal))
            line += (f", other voice below caller after 6 s: input {other_below_caller(mix, cal, bg, 96000):.1f} dB,"
                     f" output {other_below_caller(y[:k], cal[:k], bg[:k], 96000):.1f} dB")
        print(line, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
