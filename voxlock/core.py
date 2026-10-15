"""VoxLock on onnxruntime: isolate a whole call, or stream a live one chunk by chunk.

Streaming and file mode give the same output: only the LSTM state crosses frames, and it is carried over.
"""
from collections import deque
from pathlib import Path
import numpy as np
import onnxruntime as ort

SR, HOP, NFFT, PAD = 16000, 128, 512, 256
REPO, REVISION = "arbazsiddiqui/voxlock", "a50f4e1d109301d4b6f61dec57ebb1a61e7d8f03"
WIN2 = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(NFFT) / NFFT)).astype(np.float32) ** 2
GAIN = -0.5   # the network returns the caller at about -2x (trained with a scale-invariant loss); this restores the input level

def loud_sample(mix, sec=3.0, within=15.0):
    """The voice sample VoxLock is benchmarked with: the loudest `sec` seconds in the first `within` seconds."""
    seg = np.asarray(mix, np.float32)[: int(within * SR)]
    n = len(seg) // 160
    w = int(sec * SR / 160)
    if n <= w:
        return seg
    e = (seg[: n * 160].reshape(n, 160) ** 2).mean(1)
    st = int(np.convolve(e, np.ones(w), "valid").argmax()) * 160
    return seg[st: st + int(sec * SR)]

def load_audio(path):
    """Any file soundfile reads, as 16 kHz mono float32."""
    import soundfile as sf, soxr
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(1)
    return (soxr.resample(x, sr, SR) if sr != SR else x).astype(np.float32)

PRECISIONS = {                             # name: (enroll file, chunk file) in onnx/
    "int8": ("enroll", "chunk_int8"),         # LSTMs in 8-bit
    "int8-full": ("enroll", "chunk_int8_full"),  # LSTMs and matrix multiplies in 8-bit
    "fp16": ("enroll_fp16", "chunk_fp16"),    # float16 weights
    "fp32": ("enroll", "chunk"),
}

def download(precision="int8"):
    """Fetch the ONNX files from Hugging Face into its cache, once. Returns the folder holding them."""
    from huggingface_hub import hf_hub_download
    hf_hub_download(REPO, "config.yaml", revision=REVISION)
    for name in PRECISIONS[precision]:
        path = hf_hub_download(REPO, f"onnx/{name}.onnx", revision=REVISION)
    return Path(path).parent

class VoxLock:
    def __init__(self, model_dir=None, precision="int8", threads=0, fp16=None):
        """model_dir: a folder with the ONNX files; None downloads them from Hugging Face.
        precision: "int8" (75 MB), "int8-full" (67 MB, fastest), "fp16" (48 MB) or "fp32" (95 MB); see PRECISIONS.
        threads=0 lets onnxruntime choose. fp16=True/False is the 1.0 spelling of precision="fp16"/"fp32"."""
        if fp16 is not None:
            precision = "fp16" if fp16 else "fp32"
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {', '.join(PRECISIONS)}")
        model_dir = Path(model_dir) if model_dir else download(precision)
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        names = PRECISIONS[precision]
        if not (model_dir / f"{names[1]}.onnx").exists():
            names = ("enroll", "chunk")
        self.enroll = ort.InferenceSession(str(model_dir / f"{names[0]}.onnx"), so)
        self.chunk = ort.InferenceSession(str(model_dir / f"{names[1]}.onnx"), so)

    def stream(self, sample):
        """A live stream locked to the voice in `sample` (about 3 s of the caller, 16 kHz mono)."""
        return Stream(self, np.asarray(sample, np.float32))

    def live(self, sec=3.0):
        """A live stream that finds the caller by itself: it passes audio through until it has heard `sec` seconds
        of speech, takes those as the voice sample, and isolates from then on."""
        return LiveStream(self, sec)

    def isolate(self, audio, sample=None):
        """The whole call in, the caller out, same length. Without `sample`, uses loud_sample(audio)."""
        audio = np.asarray(audio, np.float32)
        s = self.stream(loud_sample(audio) if sample is None else sample)
        return np.concatenate([s.push(audio, frames_per_run=100), s.flush()])[: len(audio)]

class Stream:
    def __init__(self, vl, sample):
        self.vl = vl
        self.mag, self.ctx = vl.enroll.run(None, {"enroll": sample[None]})
        self.h = np.zeros((6, 32, 256), np.float32); self.c = np.zeros_like(self.h)
        self.head = np.zeros(0, np.float32)        # first samples, until the start padding can be built
        self.buf = None                            # padded input not yet consumed by a frame
        self.ola = np.zeros(NFFT, np.float32); self.env = np.zeros(NFFT, np.float32)
        self.dropped = 0                           # leading padded samples still to drop from the output
        self.n_in = 0

    def _run(self, frames_per_run):
        out = []
        while self.buf is not None and len(self.buf) >= NFFT:
            k = min(frames_per_run, 1 + (len(self.buf) - NFFT) // HOP)
            seg = self.buf[: (k - 1) * HOP + NFFT]
            fr, self.h, self.c = self.vl.chunk.run(None, {"wav": seg[None], "enroll_mag": self.mag, "context": self.ctx,
                                                          "h": self.h, "c": self.c})
            ola = np.concatenate([self.ola, np.zeros(k * HOP, np.float32)])
            env = np.concatenate([self.env, np.zeros(k * HOP, np.float32)])
            for j in range(k):
                ola[j * HOP: j * HOP + NFFT] += fr[0, j]; env[j * HOP: j * HOP + NFFT] += WIN2
            done = k * HOP                         # samples no later frame will touch
            y = ola[:done] / np.where(env[:done] > 1e-11, env[:done], 1.0)
            self.ola, self.env = ola[done:], env[done:]
            self.buf = self.buf[done:]
            cut = min(self.dropped, len(y)); self.dropped -= cut
            out.append(y[cut:] * GAIN)
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def push(self, chunk, frames_per_run=100):
        """Any number of 16 kHz samples in; the isolated audio that is ready comes out, 24 ms behind the input."""
        chunk = np.asarray(chunk, np.float32)
        self.n_in += len(chunk)
        if self.buf is None:
            self.head = np.concatenate([self.head, chunk])
            if len(self.head) <= PAD:
                return np.zeros(0, np.float32)
            self.buf = np.concatenate([self.head[1: PAD + 1][::-1], self.head])  # torch.stft centre padding (reflect)
            self.dropped = PAD
        else:
            self.buf = np.concatenate([self.buf, chunk])
        return self._run(frames_per_run)

    def flush(self):
        """End of the call: returns the last samples still held back."""
        if self.buf is None:
            return np.zeros(0, np.float32)
        x = np.concatenate([self.buf, self.buf[-PAD - 1: -1][::-1]])  # end padding (reflect)
        nfr = 1 + (len(x) - NFFT) // HOP
        x = np.pad(x, (0, max(0, (nfr - 1) * HOP + NFFT - len(x))))
        self.buf = x
        y = self._run(100)
        y = np.concatenate([y, self.ola / np.where(self.env > 1e-11, self.env, 1.0) * GAIN])
        self.buf = None
        return y

class LiveStream:
    """Pass-through until `sec` seconds of speech are heard, then a Stream locked to that speech.

    Speech = 10 ms frames at least `margin` dB above the noise floor, the quietest frame of the last 2 s. A window
    minimum keeps finding the gaps between words, so a caller who talks without pausing still fills the sample.
    Digital silence (below -90 dBFS, e.g. before a microphone opens) is left out of the floor, or the line noise
    after it would count as speech.
    At the switch the stream is first run over the last 250 ms of audio, whose output is dropped, so it does not start
    from an empty LSTM state, and the isolated audio fades in over 10 ms."""

    WARM, FADE, FLOOR = 4000, 160, 200

    def __init__(self, vl, sec=3.0, margin=10.0):
        self.vl, self.need, self.margin = vl, int(sec * SR / 160), margin
        self.stream = None
        self.voiced, self.energy = [], deque(maxlen=self.FLOOR)
        self.rest = np.zeros(0, np.float32)        # samples not yet in a whole 10 ms frame
        self.recent = np.zeros(0, np.float32)      # the last WARM input samples
        self.skip = 0                              # stream output samples still to drop (the warm-up)
        self.after = np.zeros(0, np.float32)       # input since the switch, for the fade-in
        self.faded = 0

    @property
    def locked(self):
        return self.stream is not None

    def _isolated(self, chunk, frames_per_run):
        if self.faded < self.FADE:
            self.after = np.concatenate([self.after, chunk])
        y = self.stream.push(chunk, frames_per_run)
        cut = min(self.skip, len(y)); self.skip -= cut; y = y[cut:]
        if self.faded < self.FADE and len(y):
            n = min(len(y), self.FADE - self.faded)
            r = (np.arange(self.faded, self.faded + n, dtype=np.float32) + 1) / self.FADE
            y = y.copy(); y[:n] = r * y[:n] + (1 - r) * self.after[self.faded: self.faded + n]
            self.faded += n
            if self.faded >= self.FADE: self.after = np.zeros(0, np.float32)
        return y

    def push(self, chunk, frames_per_run=100):
        chunk = np.asarray(chunk, np.float32)
        if self.stream is not None:
            return self._isolated(chunk, frames_per_run)
        self.recent = np.concatenate([self.recent, chunk])[-self.WARM:]
        x = np.concatenate([self.rest, chunk])
        n = len(x) // 160
        for f in x[: n * 160].reshape(n, 160):
            e = 10 * np.log10((f ** 2).mean() + 1e-10)
            if e > -90:
                self.energy.append(e)
            if len(self.energy) >= 20 and e >= min(self.energy) + self.margin:
                self.voiced.append(f)
        self.rest = x[n * 160:]
        if len(self.voiced) >= self.need:
            self.stream = self.vl.stream(np.concatenate(self.voiced[: self.need]))
            self.voiced = []; self.energy.clear()
            self.skip = len(self.recent)
            self._isolated_warm(self.recent, frames_per_run)
        return chunk

    def _isolated_warm(self, warm, frames_per_run):
        y = self.stream.push(warm, frames_per_run)
        cut = min(self.skip, len(y)); self.skip -= cut

    def flush(self):
        return self.stream.flush() if self.stream is not None else np.zeros(0, np.float32)
