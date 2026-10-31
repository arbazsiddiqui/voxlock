"""VoxLock as a LiveKit Agents noise-cancellation processor: keeps the caller's voice and removes the other voices in
the room before VAD, turn detection and speech-to-text.

    from livekit.agents import room_io
    from voxlock.livekit import VoxLockFilter

    await session.start(
        agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(sample_rate=16000, noise_cancellation=VoxLockFilter()),
        ),
    )

Install with `pip install "voxlock[livekit]"`.
"""

from __future__ import annotations

import logging
import sys
import threading

import numpy as np

from livekit import rtc

from . import __version__
from .core import download
from .realtime import Caller, preload

__all__ = ["VoxLockFilter", "preload"]

logger = logging.getLogger("voxlock.livekit")


def _pre_connect() -> bool:
    """Whether LiveKit is replaying the audio the caller recorded before the agent joined: RoomIO runs those frames
    through the processor in one burst, at the client's rate, outside the track's stream."""
    f = sys._getframe(2)                                      # 0: here, 1: _process
    for _ in range(4):                                        # a few levels up, for subclasses that wrap _process
        if f is None or f.f_code.co_name == "_run":           # AudioStream._run: the track's live frames
            return False
        if f.f_code.co_name == "_apply_audio_processor":
            return True
        f = f.f_back
    return False


class VoxLockFilter(rtc.FrameProcessor[rtc.AudioFrame]):
    """LiveKit frame processor that isolates the caller's voice with VoxLock.

    Pass it as `room_io.AudioInputOptions(noise_cancellation=VoxLockFilter())`, one per session. Each frame comes back
    at the size it came in, a fixed delay behind: two frames plus 24 ms at a 16 kHz session (124 ms with LiveKit's
    50 ms frames), plus up to 39 ms for resampling at 24 or 48 kHz. The model runs on its own thread, so the agent's
    event loop never waits for it; a stretch the model has not finished in time goes out unprocessed.

    VoxLock needs about 3 seconds of the caller's voice before it isolates. Without a `voice_sample`, it passes audio
    through until it has heard 3 seconds of speech, takes those as the sample, and isolates from then on. The lock
    holds when the caller's track changes (a microphone toggle); a different participant starts over.
    """

    def __init__(
        self,
        *,
        voice_sample: np.ndarray | None = None,
        sample_seconds: float = 3.0,
        threads: int = 1,
        model_dir: str | None = None,
        precision: str = "int8",
        delay_ms: float | None = None,
    ) -> None:
        """Initialize the VoxLock filter.

        Args:
            voice_sample: At least 3 seconds of the caller's voice, 16 kHz mono float32; a longer sample gives its
                loudest 3 s in the first 30 s. When given, isolation starts from the first audio frame.
            sample_seconds: Seconds of speech to collect as the voice sample when `voice_sample` is not given, at
                least 3.
            threads: onnxruntime threads. One is enough for a call: the int8 model takes about 70% of a server core
                with LiveKit's 50 ms frames.
            model_dir: Folder with the ONNX files. None uses the Hugging Face cache, filled by `voxlock download` or
                on first use (74 MB). The model is loaded once per process and shared by every call (see `preload`).
            precision: "int8", "int8-full", "fp16" or "fp32" (the benchmarked model).
            delay_ms: How far the output trails the input. None is two frames plus 24 ms, plus the resampling delay
                when the session is not at 16 kHz: the model gets a frame's time to finish each frame. Shorter leaves
                stretches unprocessed when the CPU is busy.
        """
        self._voice_sample = voice_sample
        self._sample_seconds = sample_seconds
        self._threads = threads
        self._model_dir = model_dir
        self._precision = precision
        self._delay_ms = delay_ms
        self._enabled = True
        self._caller: Caller | None = None
        self._n = 0
        self._identity: str | None = None
        self._track: str | None = None
        self._restart = True

    @property
    def enabled(self) -> bool:
        """Whether LiveKit runs the frames through VoxLock."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        if value and not self._enabled:                         # LiveKit skipped the frames in between
            self._restart = True
        self._enabled = value

    @property
    def delay_ms(self) -> float | None:
        """How far the output trails the input, once audio has arrived."""
        c = self._caller
        return 1000 * c.delay / c.rate if c is not None and c.rate else None

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        if not self._enabled or _pre_connect():
            return frame
        rate, ch, n = frame.sample_rate, frame.num_channels, frame.samples_per_channel
        if self._caller is None:
            self._caller = Caller(voice_sample=self._voice_sample, sample_seconds=self._sample_seconds,
                                  threads=self._threads, model_dir=self._model_dir, precision=self._precision,
                                  delay_ms=self._delay_ms)
        c = self._caller
        if self._restart or rate != c.rate or n != self._n:
            self._restart, self._n = False, n
            c.start_segment(rate, n)
        pcm = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768
        x = pcm if ch == 1 else pcm.reshape(-1, ch).mean(axis=1)
        y = (np.clip(c.process(x), -1, 1) * 32767).astype(np.int16)
        if ch > 1:
            y = np.repeat(y, ch)
        return rtc.AudioFrame(data=y.tobytes(), sample_rate=rate, num_channels=ch, samples_per_channel=n)

    def _on_stream_info_updated(self, *, room_name: str, participant_identity: str, publication_sid: str) -> None:
        if participant_identity != self._identity:              # another participant: another caller
            self._identity = participant_identity
            self._drop_caller()
        if publication_sid != self._track:
            self._track = publication_sid
            self._restart = True

    def _on_stream_info_cleared(self) -> None:
        self._track = None
        self._restart = True

    def _drop_caller(self) -> None:
        if self._caller is not None:
            self._caller.close()
        self._caller = None

    def _close(self) -> None:
        self._drop_caller()


def _register() -> None:
    from livekit.agents import Plugin

    class VoxLockPlugin(Plugin):
        def __init__(self) -> None:
            super().__init__("voxlock", __version__, "voxlock", logger)

        def download_files(self) -> None:
            download("int8")

    registered = any(p.package == "voxlock" for p in Plugin.registered_plugins)
    if not registered and threading.current_thread() is threading.main_thread():
        Plugin.register_plugin(VoxLockPlugin())


_register()
