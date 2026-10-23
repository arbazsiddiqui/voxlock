# Changelog

## 1.1.2

- A voice sample of any length from 3 seconds up works: a longer sample gives its loudest 3 seconds in the first 30. `stream()`, `isolate(sample=)` and the Pipecat filter's `voice_sample=` used to fail on anything but exactly 3 seconds.
- `isolate()` returns a call shorter than 3 seconds unchanged, with a warning, instead of failing. A sample under 3 seconds, or `live(sec=)` under 3, is refused with a clear error.

## 1.1.1

- `live()` and the Pipecat filter no longer stall when the caller talks without pausing.
- Digital silence at the start of a call no longer puts line noise into the voice sample.

## 1.1.0

- 8-bit models: `VoxLock(precision=...)` with `int8` (the new default), `int8-full`, `fp16` and `fp32`; `voxlock --precision`.
  `int8` scores 5.11% word error rate on the Krisp benchmark against 4.95% for `fp32`, and streams in real time on fewer CPU threads.
- `VoxLock.live()`: a stream that takes the first 3 seconds of speech as the voice sample, passing the audio through until then.
- Pipecat audio filter, `pip install "voxlock[pipecat]"`: `voxlock.pipecat.VoxLockFilter`. Tested with Pipecat 1.12.0.
- Output is at the input's level. 1.0.0 returned the caller at twice the level, so the command line could clip loud words.

## 1.0.0

- First release: `VoxLock.isolate`, `VoxLock.stream` and the `voxlock` command line.
