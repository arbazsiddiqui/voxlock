"""VoxLockFilter end to end through a LiveKit server: a caller publishes a recording over WebRTC (Opus), an agent
receives it with the filter, either on an rtc.AudioStream or inside an AgentSession's RoomIO. Checks that LiveKit
calls the filter, that every frame comes back whole, and that the other voice is removed from what the agent hears.

    livekit-server --dev &
    python tests/e2e_livekit.py --mode session     (or --mode stream)
"""
import argparse, asyncio, time
from pathlib import Path

import numpy as np

from livekit import api, rtc

import voxlock
from voxlock.livekit import VoxLockFilter

MIX = Path(__file__).resolve().parent.parent / "demo" / "audio" / "bystander_000_off.mp3"
URL, KEY, SECRET, ROOM = "ws://localhost:7880", "devkey", "secret", "voxlock-e2e"


class Recording(VoxLockFilter):
    """The filter, keeping what LiveKit hands it and what it gives back."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.ins, self.outs, self.info = [], [], []

    def _process(self, frame):
        out = super()._process(frame)
        self.ins.append(np.frombuffer(frame.data, np.int16).copy())
        self.outs.append(np.frombuffer(out.data, np.int16).copy())
        assert out.samples_per_channel == frame.samples_per_channel
        self.delay, self.missed = self.delay_ms, self._caller.missed / frame.sample_rate
        return out

    def _on_stream_info_updated(self, **kw):
        self.info.append(kw)
        super()._on_stream_info_updated(**kw)


def token(identity):
    return (api.AccessToken(KEY, SECRET).with_identity(identity)
            .with_grants(api.VideoGrants(room_join=True, room=ROOM)).to_jwt())


async def publish(mix):
    room = rtc.Room()
    await room.connect(URL, token("caller"))
    source = rtc.AudioSource(16000, 1)
    track = rtc.LocalAudioTrack.create_audio_track("mic", source)
    await room.local_participant.publish_track(track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
    await asyncio.sleep(1.0)
    pcm = (np.clip(mix, -1, 1) * 32767).astype(np.int16)
    for i in range(0, len(pcm) - 159, 160):                  # 10 ms frames, paced by the source
        await source.capture_frame(rtc.AudioFrame(pcm[i: i + 160].tobytes(), 16000, 1, 160))
    await asyncio.sleep(1.5)
    await room.disconnect()


async def agent_stream(filt, done):
    room = rtc.Room()
    streams = []

    @room.on("track_subscribed")
    def on_track(track, *_):
        s = rtc.AudioStream.from_track(track=track, sample_rate=16000, frame_size_ms=50, noise_cancellation=filt)
        streams.append(asyncio.ensure_future(drain(s)))

    async def drain(s):
        async for _ in s:
            pass

    await room.connect(URL, token("agent"))
    await done.wait()
    await room.disconnect()


async def agent_session(filt, done):
    from livekit.agents import Agent, AgentSession, room_io

    room = rtc.Room()
    await room.connect(URL, token("agent"))
    session = AgentSession()
    await session.start(
        Agent(instructions="Listen."),
        room=room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(sample_rate=16000, noise_cancellation=filt),
            audio_output=False, text_output=False,
        ),
    )
    await done.wait()
    await session.aclose()
    await room.disconnect()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["stream", "session"], default="session")
    a = ap.parse_args()
    mix = voxlock.load_audio(MIX)
    filt = Recording()
    done = asyncio.Event()
    agent = asyncio.ensure_future((agent_session if a.mode == "session" else agent_stream)(filt, done))
    await asyncio.sleep(2.0)
    t = time.time()
    await publish(mix)
    done.set()
    await agent
    x = np.concatenate(filt.ins).astype(np.float32) / 32768
    y = np.concatenate(filt.outs).astype(np.float32) / 32768
    d = round(filt.delay * 16)
    lag = max(range(0, 16000 * 3, 16), key=lambda k: np.dot(x[k: k + 32000], mix[:32000]))   # where the mix starts
    other = slice(lag + 10 * 16000 + 3200, lag + 13 * 16000 + 12800)    # only the other voice talks here
    xo, yo = x[other], y[other.start + d: other.stop + d]
    below = 10 * np.log10((xo @ xo) / (yo @ yo + 1e-12))
    print(f"{a.mode}: {len(filt.outs)} frames of {len(filt.ins[0])} samples, stream info {[i['participant_identity'] for i in filt.info]}, "
          f"delay {filt.delay:.0f} ms, other voice {below:.1f} dB below what LiveKit delivered, "
          f"unprocessed after the lock {filt.missed:.2f} s, {time.time() - t:.0f} s")
    assert filt.info and below > 30


if __name__ == "__main__":
    asyncio.run(main())
