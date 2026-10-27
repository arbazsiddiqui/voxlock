"""Speed on a GPU: the package's own loop with onnxruntime's CUDA provider (needs onnxruntime-gpu). Reports the
real-time factor of a whole recording (isolate) and of a live stream pushed in fixed chunks.

    python bench/gpu.py --models onnx/ --precision fp32 --audio 'ami/*/mix.wav'
"""
import argparse, functools, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import voxlock.core as core
from cpu import call_audio

core.ort.preload_dlls()                      # CUDA and cuDNN from the nvidia-* pip packages
core.ort.InferenceSession = functools.partial(core.ort.InferenceSession, providers=["CUDAExecutionProvider"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True); ap.add_argument("--precision", default="fp32")
    ap.add_argument("--audio", required=True); ap.add_argument("--sec", type=float, default=60)
    ap.add_argument("--chunk-ms", default="20,40,80")
    a = ap.parse_args()
    audio = call_audio(a.audio, a.sec)
    vl = core.VoxLock(a.models, precision=a.precision)
    print(vl.chunk.get_providers()[0], f"{len(audio) / core.SR:.0f} s of audio")
    vl.isolate(audio[: core.SR * 5])
    t = time.perf_counter(); vl.isolate(audio); rtf = (time.perf_counter() - t) / (len(audio) / core.SR)
    print(f"whole recording: rtf {rtf:.4f}")
    for ms in map(int, a.chunk_ms.split(",")):
        step = core.SR * ms // 1000
        s = vl.stream(core.loud_sample(audio))
        t = time.perf_counter()
        for i in range(0, len(audio), step):
            s.push(audio[i: i + step])
        print(f"live, {ms} ms chunks: rtf {(time.perf_counter() - t) / (len(audio) / core.SR):.3f}", flush=True)


if __name__ == "__main__":
    main()
