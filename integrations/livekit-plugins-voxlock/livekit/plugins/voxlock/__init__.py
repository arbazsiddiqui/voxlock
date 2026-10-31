"""VoxLock for LiveKit Agents: keeps the caller's voice and removes the other voices in the room before VAD, turn
detection and speech-to-text. On the agent's CPU, no API key.

    from livekit.agents import room_io
    from livekit.plugins import voxlock

    room_options=room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(sample_rate=16000, noise_cancellation=voxlock.VoxLockFilter()),
    )

The filter lives in the `voxlock` package (`voxlock.livekit`); this package puts it where LiveKit looks for plugins,
so `download-files` fetches the model.
"""

from voxlock import __version__
from voxlock.livekit import VoxLockFilter, preload

__all__ = ["VoxLockFilter", "preload", "__version__"]
