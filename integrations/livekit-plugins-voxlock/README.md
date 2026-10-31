# livekit-plugins-voxlock

[VoxLock](https://github.com/arbazsiddiqui/voxlock) for [LiveKit Agents](https://docs.livekit.io/agents/): keeps the caller's voice and removes the other voices in the room before VAD, turn detection and speech-to-text. It runs on the agent's CPU, with no server and no API key. On the [Krisp Voice Isolation Benchmark](https://huggingface.co/datasets/Krisp-AI/VoiceIsolation-Benchmark-Dataset) it has the lowest word error rate of every model tested, paid or open: 4.95%.

## Install

```bash
pip install livekit-plugins-voxlock
python my_agent.py download-files      # the 74 MB model into the Hugging Face cache
```

## Use

```python
from livekit.agents import AgentSession, room_io
from livekit.plugins import voxlock

await session.start(
    agent,
    room=ctx.room,
    room_options=room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(sample_rate=16000, noise_cancellation=voxlock.VoxLockFilter()),
    ),
)
```

`VoxLockFilter` is the same class as `voxlock.livekit.VoxLockFilter` in the `voxlock` package (`pip install "voxlock[livekit]"`). The [VoxLock README](https://github.com/arbazsiddiqui/voxlock#livekit) has the details: how it finds the caller, its delay and CPU, and a full example agent.

## License

Apache-2.0.
