"""ElevenLabs Voice Isolator (POST /v1/audio-isolation) on picked Krisp benchmark clips. Output resampled to 16 kHz,
delay removed by cross-correlation with the input, written as out_eleven.wav next to mix.wav.

    ELEVENLABS_API_KEY=... python elevenlabs_run.py krisp16 elevenlabs_clips.json
"""
import io, json, os, subprocess, sys
from pathlib import Path
import numpy as np, requests, soundfile as sf
ROOT = Path(sys.argv[1]); picks = json.load(open(sys.argv[2])); key = os.environ["ELEVENLABS_API_KEY"]
for i in picks:
    d = ROOT / i; out = d / "out_eleven.wav"
    if out.exists(): print(i, "skip", flush=True); continue
    x, sr = sf.read(d / "mix.wav", dtype="float32")
    r = requests.post("https://api.elevenlabs.io/v1/audio-isolation", headers={"xi-api-key": key},
                      files={"audio": ("mix.wav", open(d / "mix.wav", "rb"), "audio/wav")}, timeout=600)
    if r.status_code != 200: print(i, "error", r.status_code, r.text[:300], flush=True); break
    dec = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "-", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                         input=r.content, capture_output=True, check=True).stdout
    y = np.frombuffer(dec, np.float32).copy()
    n = 1 << int(np.ceil(np.log2(len(x) + len(y))))
    cc = np.fft.irfft(np.fft.rfft(y, n) * np.conj(np.fft.rfft(x, n)), n)
    lag = int(np.argmax(np.abs(np.r_[cc[-4000:], cc[:4000]]))) - 4000
    y = y[lag:] if lag > 0 else np.r_[np.zeros(-lag, np.float32), y]
    y = np.r_[y, np.zeros(max(0, len(x) - len(y)), np.float32)][: len(x)]
    sf.write(out, y, 16000)
    print(i, "ok", f"{len(x)/16000:.1f}s", "lag", lag, "fmt", r.headers.get("content-type"), flush=True)
