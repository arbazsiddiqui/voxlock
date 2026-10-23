"""VoxLock as a Pipecat audio filter: keeps the caller's voice and removes the other voices in the room before
VAD and speech-to-text.

    from pipecat.transports.base_transport import TransportParams
    from voxlock.pipecat import VoxLockFilter

    params = TransportParams(audio_in_enabled=True, audio_in_filter=VoxLockFilter())

Install with `pip install "voxlock[pipecat]"`.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from loguru import logger

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame

from .core import SR, VoxLock

try:
    import soxr
except ModuleNotFoundError as e:
    raise ImportError('VoxLockFilter needs soxr: pip install "voxlock[pipecat]"') from e


class VoxLockFilter(BaseAudioFilter):
    """Audio filter that isolates the caller's voice with VoxLock.

    VoxLock needs about 3 seconds of the caller's voice before it isolates. Without a `voice_sample`, the filter
    passes audio through unchanged until it has heard 3 seconds of speech, takes those as the sample, and isolates
    from then on. Audio is resampled to 16 kHz for the model and back to the transport rate.
    """

    def __init__(
        self,
        *,
        voice_sample: np.ndarray | None = None,
        sample_seconds: float = 3.0,
        chunk_ms: int = 80,
        threads: int = 2,
        model_dir: str | None = None,
        precision: str = "int8",
    ) -> None:
        """Initialize the VoxLock filter.

        Args:
            voice_sample: At least 3 seconds of the caller's voice, 16 kHz mono float32; a longer sample gives its
                loudest 3 s in the first 30 s. When given, isolation starts from the first audio frame.
            sample_seconds: Seconds of speech to collect as the voice sample when `voice_sample` is not given, at
                least 3.
            chunk_ms: Audio is run through the model in chunks of this length. With the int8 model, 80 ms runs in
                real time on 2 server CPU threads and 40 ms on 4; larger chunks use less CPU and add latency.
            threads: onnxruntime threads for this filter.
            model_dir: Folder with the ONNX files. None downloads them from Hugging Face on first use (75 MB).
            precision: "int8", "int8-full" (fastest), "fp16" or "fp32" (the benchmarked model).
        """
        self._voice_sample = voice_sample
        self._sample_seconds = sample_seconds
        self._chunk = int(SR * chunk_ms / 1000)
        self._threads = threads
        self._model_dir = model_dir
        self._precision = precision

        self._filtering = True
        self._sample_rate = 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voxlock")
        self._vl: VoxLock | None = None
        self._stream = None
        self._to16k = None
        self._from16k = None
        self._pending = np.zeros(0, np.float32)

    async def start(self, sample_rate: int):
        """Load the model and set up resampling for the transport's sample rate.

        Args:
            sample_rate: The sample rate of the input transport in Hz.
        """
        self._sample_rate = sample_rate
        loop = asyncio.get_running_loop()
        self._vl = await loop.run_in_executor(
            self._executor,
            lambda: VoxLock(self._model_dir, precision=self._precision, threads=self._threads),
        )
        self._reset()
        logger.debug(f"VoxLock filter ready at {sample_rate} Hz")

    def _reset(self):
        if self._voice_sample is not None:
            self._stream = self._vl.stream(self._voice_sample)
        else:
            self._stream = self._vl.live(sec=self._sample_seconds)
        if self._sample_rate != SR:
            self._to16k = soxr.ResampleStream(self._sample_rate, SR, 1, dtype="float32")
            self._from16k = soxr.ResampleStream(SR, self._sample_rate, 1, dtype="float32")
        self._pending = np.zeros(0, np.float32)

    async def stop(self):
        """Release the model."""
        self._stream = None
        self._vl = None
        self._executor.shutdown(wait=False)

    async def process_frame(self, frame: FilterControlFrame):
        """Process control frames to enable or disable filtering.

        Args:
            frame: The control frame containing filter commands.
        """
        if isinstance(frame, FilterEnableFrame):
            self._filtering = frame.enable

    async def filter(self, audio: bytes) -> bytes:
        """Isolate the caller's voice in a chunk of 16-bit PCM audio.

        Audio is collected until a full chunk is ready, so some calls return fewer bytes than they received
        (or none); the input transport skips empty results.

        Args:
            audio: Raw 16-bit mono PCM at the transport's sample rate.

        Returns:
            Isolated 16-bit mono PCM at the transport's sample rate.
        """
        if not self._filtering or self._stream is None:
            return audio

        x = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768
        if self._to16k is not None:
            x = self._to16k.resample_chunk(x)
        self._pending = np.concatenate([self._pending, x])
        if len(self._pending) < self._chunk:
            return b""

        chunk, self._pending = self._pending, np.zeros(0, np.float32)
        loop = asyncio.get_running_loop()
        y = await loop.run_in_executor(self._executor, self._stream.push, chunk)
        if self._from16k is not None:
            y = self._from16k.resample_chunk(y)
        return (np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes()
