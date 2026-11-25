import asyncio
import contextlib
import logging
import secrets
import types

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.bank import BankGenerator
from app.cache import cached_for
from app.director import load_director
from app import llm
from app.generate import llm_stack
from app.live import AnnouncedSim, BankOnlySafety, ButtonSource, WebSocketPlayer, git_sha, load_sessions
from app.log import SESSIONS, SessionLog, read_session
from app.pauses import Pauses
from app.protocol import ProtocolError, check_pause, parse
from app.recording import Recorder
from app.rooms import REACTION, RealClock
from app.show import UniformSelector, run_show

log = logging.getLogger(__name__)
app = FastAPI(title="humor-me")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
tts_engine = None  # loading Kokoro takes seconds, so the first show pays for it, not server start
sessions_dir = SESSIONS
reaction_config = REACTION
HEALTH_MIN = 0.99
AUDIO_WAIT_S = 15.0  # the operator may be answering the browser's microphone prompt


def get_tts():
    global tts_engine
    if tts_engine is None:
        from app.tts import KokoroTTS, SerialTTS

        tts_engine = SerialTTS(KokoroTTS())
    return tts_engine


def make_stack(room: str):
    """(generator, safety, fallback, bridge lines). `JOKE_SOURCE=llm` brings the real model and filter; otherwise the
    bank alone, which needs no key or quota (and is what the tests run)."""
    if llm.setting("JOKE_SOURCE", "bank") == "llm":
        return llm_stack(room)
    return BankGenerator(), BankOnlySafety(), None, None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


async def _release(ws: WebSocket, live: types.SimpleNamespace, reason: str | None = None) -> None:
    """End the recording, once: later frames are dropped without comment, and the browser is told to stop its mic."""
    recorder, live.recorder = live.recorder, None
    if recorder is None:
        return
    live.late = True
    recorder.close()
    if reason:  # the browser's notice is transient; the server log keeps why a consented show went unrecorded
        log.warning("recording ended: %s", reason)
    with contextlib.suppress(Exception):
        await ws.send_json({"type": "record_stop", "reason": reason})


async def _open_recorder(ws: WebSocket, live: types.SimpleNamespace, log: SessionLog) -> dict:
    """The show waits for the first audio frame so `start_at` is known before `show_start` is written."""
    recorder = live.recorder = Recorder(log.path.with_suffix(".wav"))
    await ws.send_json({"type": "record"})
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(recorder.started.wait(), AUDIO_WAIT_S)
    if recorder.start_at is None:
        await _release(ws, live, recorder.reason or "no microphone audio arrived")
        return {"consent": True, "path": None, "start_at": None}
    return {"consent": True, "path": recorder.path.name, "start_at": recorder.start_at}


def _note_health(live: types.SimpleNamespace, msg: dict) -> None:
    """The browser's own accounting, diagnostics only (not in the session log). Capture below 99% means the microphone
    track skipped time (the file has silence there); capture time below 99% of wall time means frames are stuck in the
    browser or on the way, which the server's coverage check would also see."""
    live.health = msg
    if msg["elapsed_s"] > 0 and msg["produced_s"] / msg["elapsed_s"] < HEALTH_MIN:
        log.warning("microphone delivered %.1f s of audio in %.1f s of capture time (%.0f%%), %.0f ms filled with "
                    "silence, device rate %s Hz", msg["produced_s"], msg["elapsed_s"],
                    100 * msg["produced_s"] / msg["elapsed_s"], msg["gap_ms"], msg["rate"])
    if msg["wall_s"] > 0 and msg["elapsed_s"] / msg["wall_s"] < HEALTH_MIN:
        log.warning("capture time %.1f s in %.1f s of wall time (%.0f%%): audio is held up in the browser",
                    msg["elapsed_s"], msg["wall_s"], 100 * msg["elapsed_s"] / msg["wall_s"])


async def _audio_frame(ws: WebSocket, live: types.SimpleNamespace, data: bytes) -> None:
    recorder = live.recorder
    if recorder is None:
        if not live.late:  # a frame in flight when the recording ended is not an error
            await ws.send_json({"type": "error", "message": "audio refused: no recording was agreed for this show"})
        return
    try:
        await asyncio.to_thread(recorder.write, data)  # disk writes must not stall the show's clock
        if recorder.start_at is not None:
            recorder.started.set()  # here, not in the thread: asyncio.Event is not thread-safe
    except ValueError as e:
        await ws.send_json({"type": "error", "message": str(e)})
    except OSError as e:  # a full disk or a file locked by a sync tool must not end the show
        log.warning("recording stopped: %s", e)
        if live.recorder is recorder:
            await _release(ws, live, "the recording stopped: the disk refused the audio")


def _rows_of(live: types.SimpleNamespace) -> list[dict]:
    """The current show's log rows, [] when it never wrote a file. Blocking: call it off the event loop."""
    path = live.log.path if live.log is not None else None
    return read_session(path) if path is not None and path.exists() else []


def _ended(rows: list[dict]) -> bool:
    """A show can be rated once its log holds `show_end` (so also `show_start`): a rating row in a log with no
    `show_start` would make the next start's replay fail."""
    return any(r["event"] == "show_end" for r in rows)


async def _rate(ws: WebSocket, live: types.SimpleNamespace, value: int) -> None:
    """Append the room's rating to the show that just ended. The file itself is asked, not a flag. A failure to read
    or write is an `error` that leaves the socket open, so the operator can retry or skip."""
    try:
        rows = await asyncio.to_thread(_rows_of, live)
        if not _ended(rows):
            refusal = "rating refused: no finished show to rate"
        elif any(r["event"] == "rating" for r in rows):
            refusal = "rating refused: this show was already rated"
        else:
            await asyncio.to_thread(live.log.write, "rating", value=value)
            await ws.send_json({"type": "rating_saved", "value": value})
            return
    except (ValueError, OSError) as e:
        refusal = f"couldn't save the rating: {e}"
    await ws.send_json({"type": "error", "message": refusal})


async def _pause(ws: WebSocket, live: types.SimpleNamespace, kind: str) -> None:
    """Write the pause or resume row on the show's clock, so it lines up with `audio_start_at` and the recording. A show
    is running from just before `show_start` is written to the moment `run_show` returns or raises, so a pause during
    the microphone prompt or after `show_end` is refused. While paused the show holds: no line starts, and the reaction
    window and the hold stop their timers (`app.pauses`). Refusals carry `about: "pause"` so the browser undoes only
    its own pause."""
    try:
        check_pause(kind, running=live.running, paused=live.pauses.paused)
        now = live.clock.now()
        live.log.write(kind, **{"paused_at" if kind == "pause" else "resumed_at": now})
        (live.pauses.pause if kind == "pause" else live.pauses.resume)(now)  # after the write: a failed row changes nothing
        await ws.send_json({"type": "pause_state", "paused": kind == "pause"})
    except ProtocolError as e:
        await ws.send_json({"type": "error", "about": "pause", "message": str(e)})
    except OSError as e:
        await ws.send_json({"type": "error", "about": "pause", "message": f"{kind} not saved: {e}"})


async def _show(ws: WebSocket, msg: dict, played: asyncio.Queue, live: types.SimpleNamespace) -> None:
    seed = secrets.randbits(32)  # the director's seed, logged as policy.seed; selector and sim draw their own
    try:
        sessions, sha, tts = await asyncio.gather(asyncio.to_thread(load_sessions, sessions_dir),
                                                  asyncio.to_thread(git_sha), asyncio.to_thread(get_tts))
        director = load_director(sessions, seed=seed)
        clock = live.clock = RealClock()
        show_log = live.log = SessionLog(sessions_dir)
        audio = await _open_recorder(ws, live, show_log) if msg["consent"] else None
        if msg["room"] == "live":
            source = live.source = ButtonSource(ws.send_json, clock, live.pauses)
        else:
            source = AnnouncedSim(ws.send_json, msg["context"], secrets.randbits(32))
        generator, safety, fallback, bridge_lines = make_stack(msg["room"])
        live.running = True  # cleared with no await in between when run_show returns or raises, so a pause after show_end is refused
        try:
            await run_show(director, msg["room"], msg["context"], generator=generator, safety=safety,
                           selector=UniformSelector(secrets.randbits(32)), tts=tts,
                           player=WebSocketPlayer(ws.send_json, ws.send_bytes, played), source=source,
                           clock=clock, log=show_log, pauses=live.pauses, explore_seed=seed, operator=msg["operator"],
                           audio=audio, git_sha=sha, config=reaction_config, fallback=fallback,
                           bridge_lines=bridge_lines, cached=cached_for(bridge_lines, sessions))
        finally:
            live.running = False
        status = "finished"
    except Exception as e:  # cancellation is not caught: a closed socket has nobody to tell
        log.exception("show failed")
        status = "aborted"
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "message": f"show failed: {e}"})
    finally:
        live.source = None
        await _release(ws, live)
    try:
        ratable = _ended(await asyncio.to_thread(_rows_of, live))  # false when the show died before it wrote a log
    except (ValueError, OSError):
        ratable = False
    with contextlib.suppress(Exception):
        await ws.send_json({"type": "show_end", "status": status, "ratable": ratable})


@app.websocket("/ws")
async def session(ws: WebSocket) -> None:
    await ws.accept()
    played: asyncio.Queue = asyncio.Queue()
    live = types.SimpleNamespace(source=None, recorder=None, late=False, health=None, log=None,
                                 clock=None, pauses=Pauses(), running=False)  # late: audio was asked for and ended
    show = None
    try:
        while True:  # keeps running while the show does, so keys and "played" are heard
            frame = await ws.receive()
            if frame["type"] == "websocket.disconnect":
                break
            if frame.get("bytes") is not None:  # binary is only ever audio; every control message is a text frame
                await _audio_frame(ws, live, frame["bytes"])
                continue
            try:
                msg = parse(frame)
                if msg["type"] == "start" and show is not None and not show.done():
                    raise ProtocolError("a show is already running")
            except ProtocolError as e:
                await ws.send_json({"type": "error", "message": str(e)})
                continue
            if msg["type"] == "played":
                played.put_nowait(msg)
            elif msg["type"] == "audio_health":
                _note_health(live, msg)
            elif msg["type"] == "audio_start":
                if live.recorder is not None:
                    live.recorder.hint_start(msg["wall_ms"] / 1000)
            elif msg["type"] == "audio_failed":
                if live.recorder is not None and live.recorder.start_at is None:
                    live.recorder.abandon(msg["reason"])
                else:  # it failed after audio began: the file so far stays, the show goes on
                    await _release(ws, live, msg["reason"])
            elif msg["type"] in ("pause", "resume"):
                await _pause(ws, live, msg["type"])
            elif msg["type"] == "rating":
                await _rate(ws, live, msg["value"])
            elif msg["type"] == "reaction":
                if live.source is None:
                    log.info("ignored press %s: no live show", msg["key"])
                else:
                    live.source.press(msg["bit"], msg["joke"], msg["key"])
            else:
                live.late = False
                live.log = None  # the previous show can no longer be rated
                live.clock, live.pauses = None, Pauses()
                show = asyncio.create_task(_show(ws, msg, played, live))
    except WebSocketDisconnect:
        pass
    finally:
        if live.recorder is not None:
            live.recorder.close()
        if show is not None:
            show.cancel()  # run_show writes show_end: aborted on the way out
            await asyncio.gather(show, return_exceptions=True)
