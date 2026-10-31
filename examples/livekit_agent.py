"""A LiveKit voice agent with VoxLock on the input: other voices in the caller's room are removed before VAD, turn
detection and speech-to-text hear them.

A standard LiveKit Agents voice agent; the VoxLock lines are `noise_cancellation=VoxLockFilter()` and the model
loaded once per worker process in `prewarm`.

    pip install "voxlock[livekit]" "livekit-agents[silero]" python-dotenv
    python examples/livekit_agent.py download-files     # VoxLock and Silero into the cache
    python examples/livekit_agent.py dev                # then join the room from the Agents Playground

Needs LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET in the environment or a .env file. Speech-to-text, the LLM
and text-to-speech go through LiveKit Inference. `console` mode reads your microphone without a room, so VoxLock does
not run there: test it in a room.
"""

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, JobProcess, cli, room_io
from livekit.plugins import silero
from voxlock.livekit import VoxLockFilter, preload

load_dotenv(override=True)

server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    preload()                                    # VoxLock, shared by every call this process runs


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt="deepgram/nova-3",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-3",
    )
    await session.start(
        Agent(instructions="You are a helpful voice assistant. Keep your answers short."),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                sample_rate=16000,                   # VoxLock's rate: no resampling, the shortest delay
                noise_cancellation=VoxLockFilter(),  # one per session
            ),
        ),
    )
    await session.generate_reply(instructions="Greet the user and offer your help.")


if __name__ == "__main__":
    cli.run_app(server)
