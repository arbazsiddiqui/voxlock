"""VoxLock as a Pipecat audio filter: keeps the caller's voice and removes the other voices in the room before
VAD and speech-to-text.

    from pipecat.transports.base_transport import TransportParams
    from voxlock.pipecat import VoxLockFilter

    params = TransportParams(audio_in_enabled=True, audio_in_filter=VoxLockFilter())

Install with `pip install "voxlock[pipecat]"`.
"""

import asyncio

import numpy as np
from loguru import logger

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame

from .realtime import Caller, preload


class VoxLockFilter(BaseAudioFilter):
    """Audio filter that isolates the caller's voice with VoxLock.

    VoxLock needs about 3 seconds of the caller's voice before it isolates. Without a `voice_sample`, the filter
    passes audio through until it has heard 3 seconds of speech, takes those as the sample, and isolates from then on.
    Each frame comes back at the size it came in, a fixed delay behind (184 ms with the default 80 ms chunks at
    16 kHz). The model runs on its own thread, so the transport never waits for it; a stretch the model has not
    finished in time goes out unprocessed. Audio is resampled to 16 kHz for the model and back to the transport rate.
    """

    def __init__(
        self,
        *,
        voice_sample: np.ndarray | None = None,
        sample_seconds: float = 3.0,
        chunk_ms: int = 80,
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
            chunk_ms: Audio is run through the model in chunks of this length. With the int8 model on one server CPU
                thread, 80 ms takes 61% of the core and 40 ms 77%; larger chunks use less CPU and add latency.
            threads: onnxruntime threads for this filter.
            model_dir: Folder with the ONNX files. None downloads them from Hugging Face on first use (74 MB). The
                model is loaded once per process and shared by every call.
            precision: "int8", "int8-full", "fp16" or "fp32" (the benchmarked model).
            delay_ms: How far the output trails the input. None is two chunks plus 24 ms, plus the resampling delay
                when the transport is not at 16 kHz: the model gets a chunk's time to finish each chunk. Shorter
                leaves stretches unprocessed when the CPU is busy.
        """
        self._voice_sample = voice_sample
        self._sample_seconds = sample_seconds
        self._chunk_ms = chunk_ms
        self._threads = threads
        self._model_dir = model_dir
        self._precision = precision
        self._delay_ms = delay_ms

        self._filtering = True
        self._restart = True
        self._sample_rate = 0
        self._caller: Caller | None = None

    @property
    def delay_ms(self) -> float | None:
        """How far the output trails the input, once audio has arrived."""
        c = self._caller
        return 1000 * c.delay / c.rate if c is not None and c.rate else None

    async def start(self, sample_rate: int):
        """Load the model and start the worker for the transport's sample rate.

        Args:
            sample_rate: The sample rate of the input transport in Hz.
        """
        self._sample_rate = sample_rate
        await asyncio.get_running_loop().run_in_executor(
            None, lambda: preload(model_dir=self._model_dir, precision=self._precision, threads=self._threads)
        )
        self._caller = Caller(voice_sample=self._voice_sample, sample_seconds=self._sample_seconds,
                              threads=self._threads, model_dir=self._model_dir, precision=self._precision,
                              chunk_ms=self._chunk_ms, delay_ms=self._delay_ms)
        self._restart = True
        logger.debug(f"VoxLock filter ready at {sample_rate} Hz")

    async def stop(self):
        """Stop the worker."""
        if self._caller is not None:
            self._caller.close()
        self._caller = None

    async def process_frame(self, frame: FilterControlFrame):
        """Process control frames to enable or disable filtering.

        Args:
            frame: The control frame containing filter commands.
        """
        if isinstance(frame, FilterEnableFrame):
            if frame.enable and not self._filtering:            # the frames in between went out unfiltered
                self._restart = True
            self._filtering = frame.enable

    async def filter(self, audio: bytes) -> bytes:
        """Isolate the caller's voice in a chunk of 16-bit PCM audio.

        Args:
            audio: Raw 16-bit mono PCM at the transport's sample rate.

        Returns:
            Isolated 16-bit mono PCM at the transport's sample rate, as many bytes as came in.
        """
        if not self._filtering or self._caller is None or not audio:
            return audio
        x = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768
        if self._restart:
            self._restart = False
            self._caller.start_segment(self._sample_rate, len(x))
        y = self._caller.process(x)
        return (np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes()
