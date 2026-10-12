"""voxlock call.wav -o caller.wav"""
import argparse, sys, time
from pathlib import Path
import soundfile as sf
from . import SR, VoxLock, load_audio

def main(argv=None):
    ap = argparse.ArgumentParser(prog="voxlock", description="Keep the caller's voice, remove the other voices in the room.")
    ap.add_argument("inputs", nargs="+", help="audio files, any sample rate")
    ap.add_argument("-o", "--out", help="output file (one input) or folder (several); default: <name>.voxlock.wav")
    ap.add_argument("--sample", help="a recording of the caller's voice; default: the loudest 3 s in the first 15 s")
    ap.add_argument("--precision", choices=["int8", "int8-full", "fp16", "fp32"], default="int8",
                    help="model files: int8 (default), int8-full (fastest), fp16 or fp32 (the benchmarked model)")
    ap.add_argument("--fp32", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args(argv)
    vl = VoxLock(precision="fp32" if a.fp32 else a.precision, threads=a.threads)
    sample = load_audio(a.sample) if a.sample else None
    for src in map(Path, a.inputs):
        if a.out and len(a.inputs) == 1:
            dst = Path(a.out)
        elif a.out:
            dst = Path(a.out) / f"{src.stem}.wav"
        else:
            dst = src.with_name(f"{src.stem}.voxlock.wav")
        dst.parent.mkdir(parents=True, exist_ok=True)
        x = load_audio(src)
        t = time.time()
        sf.write(str(dst), vl.isolate(x, sample), SR)
        print(f"{src} -> {dst}  ({len(x) / SR:.1f} s in {time.time() - t:.1f} s)", file=sys.stderr)

if __name__ == "__main__":
    main()
