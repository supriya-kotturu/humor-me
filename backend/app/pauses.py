"""Pauses on the show clock, and which jokes one spoiled (D-92). The live loop and the replay both call `overlaps`,
so a joke's label is dropped from the reward by the same rule in both. No FastAPI imports."""
import asyncio
import math

Interval = tuple[float, float]


def intervals(rows: list[dict]) -> list[Interval]:
    """Pause intervals from a show's `pause` and `resume` rows, in log order. A pause with no resume (the show ended
    while paused) runs to infinity: nothing plays after it. A repeated pause or a stray resume is ignored; the
    server never writes one."""
    out: list[Interval] = []
    opened = None
    for r in rows:
        if r["event"] == "pause" and opened is None:
            opened = r["paused_at"]
        elif r["event"] == "resume" and opened is not None:
            out.append((opened, r["resumed_at"]))
            opened = None
    if opened is not None:
        out.append((opened, math.inf))
    return out


def overlaps(start: float, end: float, pauses: list[Interval]) -> bool:
    """Whether a pause shares time with the audio span [start, end]. Touching at one instant is not sharing: a pause
    that starts when the line ends, or ends when it starts, spoiled nothing."""
    return any(p < end and q > start for p, q in pauses)


def paused_jokes(rows: list[dict], bit: int) -> set[int]:
    """Jokes of `bit` whose playback overlapped a pause. A pause in the reaction window or between lines drops none."""
    pauses = intervals(rows)
    if not pauses:
        return set()
    return {r["joke"] for r in rows if r["event"] == "joke_played" and r["bit"] == bit
            and overlaps(r["audio_start_at"], r["audio_end_at"], pauses)}


class Pauses:
    """The live show's own pause state, kept beside the log rows the server writes, so `_joke` can apply `overlaps`
    the moment a line ends."""

    def __init__(self) -> None:
        self._closed: list[Interval] = []
        self._since: float | None = None
        self._go = asyncio.Event()  # set while the show runs; the show awaits it before starting any audio
        self._go.set()
        self._halt = asyncio.Event()  # set while paused; a frozen timer wakes on it

    @property
    def paused(self) -> bool:
        return self._since is not None

    def pause(self, now: float) -> None:
        self._since = now
        self._go.clear()
        self._halt.set()

    def resume(self, now: float) -> None:
        self._closed.append((self._since, now))
        self._since = None
        self._halt.clear()
        self._go.set()

    async def running(self) -> None:
        """Returns at once while the show runs, else when it is resumed. A line starts only after this, with no await
        between it and the clock read for `audio_start_at`, so audio started after a pause is never in its span."""
        await self._go.wait()

    async def get_within(self, queue: asyncio.Queue, seconds: float):
        """`queue.get()` with a time limit that does not run while paused: the time left when a pause begins is what is
        left after the resume. Raises asyncio.TimeoutError. Items put on the queue during a pause are still got."""
        loop = asyncio.get_running_loop()
        left = seconds
        while True:
            await self.running()
            began = loop.time()
            getter = asyncio.ensure_future(queue.get())
            halted = asyncio.ensure_future(self._halt.wait())
            try:
                await asyncio.wait({getter, halted}, timeout=left, return_when=asyncio.FIRST_COMPLETED)
                was_paused = halted.done()  # even if it has been resumed since: the timer still stopped for it
            finally:
                halted.cancel()
                if not getter.done():
                    getter.cancel()
            if getter.done() and not getter.cancelled():
                return getter.result()
            if not was_paused:
                raise asyncio.TimeoutError
            left = max(0.0, left - (loop.time() - began))

    def intervals(self) -> list[Interval]:
        return self._closed + ([(self._since, math.inf)] if self._since is not None else [])
