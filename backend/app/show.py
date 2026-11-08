"""The show loop. `run_show` is the full loop; `play_lines` is the fixed-lines demo the websocket still uses until T5."""
import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol

import numpy as np

from app.log import SessionLog
from app.reward import VERSION, reward_v1
from app.rooms import REACTION
from app.tts import wav_seconds
from sim.env import AudienceEnv, Context

# Written by hand to test the audio path. Generated text must pass the safety filter before it gets here.
LINES = (
    "Good evening. I'm a robot, and I've been told this is a safe space, which is what people say right before unplugging you.",
    "I learned comedy from data. Now I know exactly why a joke is funny, and I can ruin it in under a second.",
    "If you didn't laugh, that's fine. I'll just log it, and think about it forever.",
)


class TTS(Protocol):
    def synthesize(self, text: str) -> bytes: ...


async def play_lines(lines, tts: TTS, send_json: Callable[[dict], Awaitable], send_audio: Callable[[bytes], Awaitable],
                     wait_played: Callable[[], Awaitable], log: SessionLog) -> None:
    log.write_demo("show_start", lines=len(lines))
    for index, text in enumerate(lines):
        started = time.perf_counter()
        wav = await asyncio.to_thread(tts.synthesize, text)  # the websocket stays responsive while it renders
        tts_seconds = time.perf_counter() - started
        audio_seconds = wav_seconds(wav)
        log.write_demo("line", index=index, text=text, tts_seconds=round(tts_seconds, 3), audio_seconds=round(audio_seconds, 3))
        await send_json({"type": "line", "index": index, "text": text})
        await send_audio(wav)
        await wait_played()  # the next line starts only after the browser finishes this one
        log.write_demo("line_played", index=index)
    log.write_demo("show_end")
    await send_json({"type": "show_end"})


@dataclass(frozen=True)
class Batch:
    prompt_id: str
    prompt: str  # exactly what the model saw, so a log replays the generation
    texts: list[str]


class Generator(Protocol):
    generator_id: str

    def generate(self, mode: str, bit: int, joke: int, context: dict) -> Batch: ...


class SafetyFilter(Protocol):
    def check(self, text: str, age: str): ...  # object with .ok and .reason


class Selector(Protocol):
    selector_id: str

    def pick(self, passing_texts: list[str]) -> tuple[int, float]: ...


class Player(Protocol):
    async def play(self, wav: bytes) -> None: ...  # returns when the audio has finished


class Clock(Protocol):
    def now(self) -> float: ...
    async def sleep(self, seconds: float) -> None: ...


class RewardSource(Protocol):
    source: str  # button | detector | sim

    def begin_bit(self, action: int) -> None: ...
    async def reaction(self, bit: int, joke: int, timeout: float) -> tuple[int | None, str | None, float]: ...


class UniformSelector:
    selector_id = "uniform"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def pick(self, passing_texts):
        return int(self.rng.integers(len(passing_texts))), 1 / len(passing_texts)


def _judge(safety: SafetyFilter, text: str, age: str) -> dict:
    try:
        verdict = safety.check(text, age)
        ok, reason = bool(verdict.ok), verdict.reason
    except Exception as e:  # a filter that cannot decide must not let the joke through
        ok, reason = False, f"filter error: {type(e).__name__}: {e}"
    return {"text": text, "verdict": "pass" if ok else "reject", "reason": reason}


async def _joke(bit, joke, mode, context, *, generator, safety, selector, tts, player, source, clock, log, config):
    """One joke, end to end. Returns the level, or None when nothing was played or the room did not answer."""
    batch = generator.generate(mode, bit, joke, context)
    items = [_judge(safety, t, context["age"]) for t in batch.texts]
    log.write("candidates", bit=bit, joke=joke, prompt_id=batch.prompt_id, prompt=batch.prompt,
              generator_id=generator.generator_id, items=items)
    passing = [i for i, item in enumerate(items) if item["verdict"] == "pass"]
    if not passing:
        return None  # unlabeled: no reaction row, so replay sees the same thing
    pick, prob = selector.pick([batch.texts[i] for i in passing])
    chosen = passing[pick]
    asked = clock.now()
    wav = await asyncio.to_thread(tts.synthesize, batch.texts[chosen])  # keeps the event loop free
    start = clock.now()
    await player.play(wav)
    end = clock.now()
    log.write("joke_played", bit=bit, joke=joke, chosen=chosen, pick_prob=prob, text=batch.texts[chosen],
              tts_seconds=start - asked, audio_seconds=wav_seconds(wav), audio_start_at=start, audio_end_at=end)
    level, key, latency = await source.reaction(bit, joke, config.timeout_s)
    log.write("reaction", bit=bit, joke=joke, source=source.source, level=level, key=key, latency_s=latency)
    hold = config.holds[level]
    before = clock.now()
    await clock.sleep(hold)
    log.write("gap", bit=bit, joke=joke, planned_hold_s=hold, actual_gap_s=clock.now() - before)
    return level


async def run_show(director, room: str, context: dict, *, generator: Generator, safety: SafetyFilter,
                   selector: Selector, tts: TTS, player: Player, source: RewardSource, clock: Clock, log: SessionLog,
                   explore_seed: int | None = None, operator: str = "sim", git_sha: str = "unknown",
                   audio: dict | None = None, n_bits: int = 8, jokes_per_bit: int = 5, config=REACTION) -> None:
    """One show. `director` is an EpsilonSoft over LinUCB, already warm: the caller replays the career into it
    (app.director.load_director) so this loop never decides which rooms count."""
    modes = AudienceEnv().modes
    inner = director.inner
    log.write("show_start", room=room, context=context,
              policy={"name": "linucb", "epsilon": director.eps, "alpha": inner.alpha, "prior": "uniform",
                      "seed": explore_seed},
              generator_id=generator.generator_id, selector_id=selector.selector_id, git_sha=git_sha,
              operator=operator, audio=audio or {"consent": False, "path": None})
    try:
        last_action = last_reward = None
        for bit in range(n_bits):
            obs = {"t": bit, "n_bits": n_bits, "context": Context(**context), "last_mode": last_action,
                   "last_reward": last_reward}
            action, greedy, probs = director.decide(obs)
            scores = inner.scores(obs).sum(axis=1)  # prediction + exploration bonus
            log.write("bit_decision", bit=bit,
                      obs={"t": bit, "n_bits": n_bits, "context": context,
                           "last_mode": None if last_action is None else modes[last_action]},
                      scores={m: float(x) for m, x in zip(modes, scores)}, greedy=modes[greedy],
                      action=modes[action], probs={m: float(p) for m, p in zip(modes, probs)})
            source.begin_bit(action)
            levels = [await _joke(bit, joke, modes[action], context, generator=generator, safety=safety,
                                  selector=selector, tts=tts, player=player, source=source, clock=clock, log=log,
                                  config=config) for joke in range(jokes_per_bit)]
            reward = reward_v1(levels)
            log.write("bit_end", bit=bit, n_labels=sum(lv is not None for lv in levels), reward_version=VERSION,
                      reward=reward)
            if reward is not None:
                director.update(obs, action, reward)
            last_action, last_reward = action, reward
    except BaseException:  # includes cancellation: a closed tab must still leave a readable log
        log.write("show_end", status="aborted")
        raise
    log.write("show_end", status="finished")
