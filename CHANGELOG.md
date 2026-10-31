# Changelog

## 1.3.0

- A LiveKit Agents plugin: `voxlock.livekit.VoxLockFilter`, a frame processor for `room_io.AudioInputOptions(noise_cancellation=...)` (`pip install "voxlock[livekit]"`, or `pip install livekit-plugins-voxlock` for `from livekit.plugins import voxlock`). The model runs on its own thread, so the agent's event loop spends about 0.3 ms per frame on it; each frame comes back whole, 124 ms behind in a 16 kHz session. The caller keeps their voice lock across a track change, pre-connect audio passes through unchanged, and `download-files` fetches the model. Tested end to end through a LiveKit server, in an `rtc.AudioStream` and in an `AgentSession`.
- The Pipecat filter no longer holds up the call's audio when it locks onto the caller. The one-time catch-up at the switch ran on the transport's audio loop, which waits for the filter: on a server CPU, speech-to-text and VAD heard nothing for up to 2.5 seconds, in the middle of the caller's first sentence, and the next 4 to 6 seconds arrived late. The model now runs on its own thread and every frame comes back at once, whole, a fixed delay behind (184 ms with the default 80 ms chunks at 16 kHz); a stretch the model has not finished goes out unprocessed. Same CPU, same output once caught up.
- `voxlock download` fetches the model files ahead of time, e.g. while building a Docker image.
- Same models.

## 1.2.1

- `live()` and the Pipecat filter no longer mute the caller on some calls. At the switch from pass-through to isolation they started the model on only the last 250 ms of audio; started cold in the middle of a sentence, the model could take the caller for background and keep them muted for the rest of the call (3 of the 63 Krisp phone calls). They now run the model over the call so far (up to 30 seconds) first. On the 265 Krisp recordings `live()` goes from 5.41% to 5.15% word error rate (4.95% on whole files), and on the phone calls from 8.02% to 6.07%. The catch-up is a one-time 2 to 5 seconds of work on one CPU thread at the switch. Same models.
- A stream's `flush()` no longer returns samples past the end of the input. They were the end padding divided by a near-zero window, and could be loud: a click at the end of the call. `isolate()` and the Pipecat filter were not affected.

## 1.2.0

- Less CPU, same output. The enroll model now also computes the speaker attention's keys and values from the voice sample, once per call, instead of the chunk model recomputing them for every chunk. With `int8` on one CPU thread, 40 ms chunks take 77% of a core and 80 ms chunks 61%. The models are in `onnx/v2/` on Hugging Face; 1.1 installs keep downloading `onnx/`.
- The Pipecat filter runs on one thread by default (`threads=1`), which fits a single vCPU.
- `int8-full` keeps the attention in float, so it runs as fast as `int8`.

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
