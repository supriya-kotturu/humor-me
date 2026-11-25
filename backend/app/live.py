"""The parts of a show that exist only because of a browser: button rewards, the websocket voice, the bank-only
filter, and the session loader. No FastAPI imports, so they are testable without a server."""
import asyncio
import json
import logging
import pathlib
import subprocess
from dataclasses import dataclass

from app.bank import load_bank
from app.log import KEY_LEVELS, read_session
from app.pauses import Pauses
from app.rooms import SimRoom

log = logging.getLogger(__name__)
ROOT = pathlib.Path(__file__).resolve().parents[2]


class ButtonSource:
    """Rewards from the operator's keys. The window opens when audio ends and closes at the end of the hold (or on
    timeout). A press outside it is never a `reaction` row: replay counts every row, so it is only noted in `log`."""
    source = "button"

    def __init__(self, send, clock, pauses=None):
        self.send, self.clock = send, clock
        self.pauses = Pauses() if pauses is None else pauses  # the window and the hold stop while the show is paused
        self._window: tuple[int, int] | None = None
        self._opened = 0.0
        self._presses: asyncio.Queue = asyncio.Queue()

    def begin_bit(self, action: int) -> None:
        pass

    def press(self, bit: int, joke: int, key: str) -> None:
        if self._window != (bit, joke):
            log.info("ignored press %s for bit %s joke %s: window is %s", key, bit, joke, self._window)
            return
        self._presses.put_nowait((KEY_LEVELS[key], key, self.clock.now() - self._opened))

    async def _close(self) -> None:
        bit, joke = self._window
        self._window = None  # before the await, so a press racing the send is already late
        await self.send({"type": "reaction_window", "state": "closed", "bit": bit, "joke": joke})

    async def reaction(self, bit: int, joke: int, timeout: float):
        while not self._presses.empty():
            self._presses.get_nowait()
        self._window, self._opened = (bit, joke), self.clock.now()
        await self.send({"type": "reaction_window", "state": "open", "bit": bit, "joke": joke})
        try:
            return await self.pauses.get_within(self._presses, timeout)
        except asyncio.TimeoutError:
            await self._close()
            return None, None, timeout

    async def hold(self, bit: int, joke: int, seconds: float, clock):
        if self._window is None:  # timed out: nothing to hold for
            return None
        try:
            return await self.pauses.get_within(self._presses, seconds)
        except asyncio.TimeoutError:
            await self._close()
            return None


class AnnouncedSim(SimRoom):
    """SimRoom that also tells the browser, so the operator UI can light the key the sim 'pressed'."""

    def __init__(self, send, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.send = send

    async def reaction(self, bit: int, joke: int, timeout: float):
        await self.send({"type": "reaction_window", "state": "open", "bit": bit, "joke": joke})
        level, key, latency = await super().reaction(bit, joke, timeout)
        await self.send({"type": "sim_reaction", "bit": bit, "joke": joke, "key": key})
        return level, key, latency

    async def hold(self, bit: int, joke: int, seconds: float, clock):
        await super().hold(bit, joke, seconds, clock)
        await self.send({"type": "reaction_window", "state": "closed", "bit": bit, "joke": joke})


class WebSocketPlayer:
    def __init__(self, send_json, send_bytes, played: asyncio.Queue):
        self.send_json, self.send_bytes, self.played = send_json, send_bytes, played

    async def play(self, wav: bytes, text: str) -> None:
        while not self.played.empty():
            self.played.get_nowait()  # a stray ack must not end this line early
        await self.send_json({"type": "line", "text": text})
        await self.send_bytes(wav)
        await self.played.get()


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str | None


class BankOnlySafety:
    """Passes only text that is literally in the hand-written bank. Stands in until the real filter (T8/T9)."""

    def __init__(self, bank: dict | None = None):
        self.allowed = {j["text"] for bits in (bank or load_bank()).values() for bit in bits for j in bit["jokes"]}

    def check(self, text: str, age: str) -> Verdict:
        return Verdict(True, None) if text in self.allowed else Verdict(False, "not in the hand-written bank")


def load_sessions(directory: pathlib.Path) -> list[list[dict]]:
    """Every schema 1 or 2 log in `directory`. Old demo logs (`schema: 0`) are skipped; anything else unreadable raises,
    so a corrupt log fails the start instead of quietly changing what the director learned."""
    sessions = []
    for path in sorted(directory.glob("*.jsonl")):
        first = path.read_bytes().split(b"\n", 1)[0]
        try:
            if json.loads(first).get("schema") == 0:
                continue
        except (ValueError, AttributeError):
            pass  # not a demo row: read_session decides whether it is valid
        rows = read_session(path)
        if rows:
            sessions.append(rows)
    return sessions


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5,
                             check=True)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"
