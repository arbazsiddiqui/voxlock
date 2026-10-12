"""VoxLock on onnxruntime: isolate a whole call, or stream a live one chunk by chunk.

Streaming and file mode give the same output: only the LSTM state crosses frames, and it is carried over.
"""
from pathlib import Path
import numpy as np
import onnxruntime as ort

SR, HOP, NFFT, PAD = 16000, 128, 512, 256
REPO, REVISION = "arbazsiddiqui/voxlock", "770c2e4ede56458ca0ccf977de935e46d6733f2c"
WIN2 = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(NFFT) / NFFT)).astype(np.float32) ** 2

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

def download(fp16=True):
    """Fetch the ONNX files from Hugging Face into its cache, once. Returns the folder holding them."""
    from huggingface_hub import hf_hub_download
    sfx = "_fp16" if fp16 else ""
    hf_hub_download(REPO, "config.yaml", revision=REVISION)
    for name in (f"onnx/enroll{sfx}.onnx", f"onnx/chunk{sfx}.onnx"):
        path = hf_hub_download(REPO, name, revision=REVISION)
    return Path(path).parent

class VoxLock:
    def __init__(self, model_dir=None, fp16=True, threads=0):
        """model_dir: a folder with enroll*.onnx and chunk*.onnx; None downloads them from Hugging Face.
        fp16=True uses the 50 MB float16-weight files, False the float32 ones. threads=0 lets onnxruntime choose."""
        model_dir = Path(model_dir) if model_dir else download(fp16)
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        sfx = "_fp16" if fp16 and (model_dir / "chunk_fp16.onnx").exists() else ""
        self.enroll = ort.InferenceSession(str(model_dir / f"enroll{sfx}.onnx"), so)
        self.chunk = ort.InferenceSession(str(model_dir / f"chunk{sfx}.onnx"), so)

    def stream(self, sample):
        """A live stream locked to the voice in `sample` (about 3 s of the caller, 16 kHz mono)."""
        return Stream(self, np.asarray(sample, np.float32))

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
            out.append(y[cut:])
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
        y = np.concatenate([y, self.ola / np.where(self.env > 1e-11, self.env, 1.0)])
        self.buf = None
        return y
