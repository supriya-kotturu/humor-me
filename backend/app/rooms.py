"""Where the room's reaction comes from, and the clock the show runs on (docs/plans/live-loop.md, "Reaction labels")."""
import asyncio
import time
from dataclasses import dataclass, field

from app.log import KEY_LEVELS
from sim.env import AudienceEnv, Context

KEY_FOR_LEVEL = {level: key for key, level in KEY_LEVELS.items()}


@dataclass(frozen=True)
class ReactionConfig:
    """Holds and timeout are judgment calls, not measured; a null level (timed out) holds 0 s."""
    holds: dict = field(default_factory=lambda: {3: 2.5, 2: 1.5, 1: 1.0, 0: 0.3, None: 0.0})
    timeout_s: float = 6.0


class RealClock:
    def now(self) -> float:
        return time.time()  # same timebase as the log's `at`

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class SimRoom:
    """The simulated audience as a RewardSource. One `AudienceEnv.step` per bit, served a joke at a time."""
    source = "sim"

    def __init__(self, context: dict, seed: int, latency_s: float = 0.5, env: AudienceEnv | None = None):
        self.env = env or AudienceEnv()
        self.env.reset(Context(**context), seed=seed)
        self.latency_s = latency_s  # made up: a sim room has no operator to be slow
        self._levels: list[int] = []

    def begin_bit(self, action: int) -> None:
        self._levels = self.env.step(action)[3]["levels"]

    async def reaction(self, bit: int, joke: int, timeout: float):
        level = self._levels[joke]
        return level, KEY_FOR_LEVEL[level], self.latency_s


REACTION = ReactionConfig()
