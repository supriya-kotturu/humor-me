"""The show loop: `run_show` runs one show against whatever voice, room and clock it is given."""
import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.cache import CachedBit, screen
from app.log import SessionLog
from app.pauses import Pauses, overlaps
from app.reward import VERSION, reward_v1
from app.rooms import REACTION
from app.tts import wav_seconds
from sim.env import AudienceEnv, Context

_log = logging.getLogger(__name__)


class TTS(Protocol):
    def synthesize(self, text: str) -> bytes: ...


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
    # Optionally `prefetch(texts, age)`: the show calls it once per bit, ahead of the per-line `check`s.


class Selector(Protocol):
    selector_id: str

    def pick(self, passing_texts: list[str]) -> tuple[int, float]: ...


class Player(Protocol):
    async def play(self, wav: bytes, text: str) -> None: ...  # returns when the audio has finished


class Clock(Protocol):
    def now(self) -> float: ...
    async def sleep(self, seconds: float) -> None: ...


class RewardSource(Protocol):
    source: str  # button | detector | sim

    def begin_bit(self, action: int) -> None: ...
    async def reaction(self, bit: int, joke: int, timeout: float) -> tuple[int | None, str | None, float]: ...

    async def hold(self, bit: int, joke: int, seconds: float, clock: Clock) -> tuple[int | None, str | None, float] | None:
        """Wait out the hold. Returns a corrected (level, key, latency_s) if the operator changed their mind, else None."""
        ...


class UniformSelector:
    selector_id = "uniform"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def pick(self, passing_texts):
        return int(self.rng.integers(len(passing_texts))), 1 / len(passing_texts)


def _judge(check, text: str, age: str) -> dict:
    try:
        verdict = check(text, age)
        ok, reason = bool(verdict.ok), verdict.reason
    except Exception as e:  # a filter that cannot decide must not let the joke through
        ok, reason = False, f"filter error: {type(e).__name__}: {e}"
    return {"text": text, "verdict": "pass" if ok else "reject", "reason": reason}


@dataclass
class _Plan:
    """One joke decided before it plays. `items` is the joke's one `candidates` row, written when the joke resolves:
    the generated candidates, then the bank's lines once the bank is consulted. `options` are (item index, pick_prob)
    in the order to try, so a line that will not render falls through to the next, and then to the bank."""
    items: list[dict]
    options: list[tuple[int, float]]
    prompt_id: str
    prompt: str
    generator_id: str
    bank: Callable[[], Awaitable[None]] | None = None  # consults the bank once; None when there is none or it ran
    render: asyncio.Task | None = None  # the render of options[0], started ahead of its turn


async def _render(tts, text: str, clock):
    asked = clock.now()
    wav = await asyncio.to_thread(tts.synthesize, text)  # keeps the event loop free
    return wav, clock.now() - asked


def _pick_thread(batches, passing, selector) -> tuple[int, list[int]] | None:
    """Candidate i of every joke belongs to thread i (the generator is prompted so), and a bit plays one thread, so
    the closer's callback matches the opener the room heard. Picks uniformly among the threads with the most passing
    jokes: every full thread when any passes, else the best partial ones. Returns the pick and the tied threads
    (the pick's probability is 1 over their number); None when no candidate passed anywhere."""
    live = [j for j, b in enumerate(batches) if b]
    if not live:
        return None
    n = min(len(batches[j].texts) for j in live)
    score = [sum(t in passing[j] for j in live) for t in range(n)]
    tied = [t for t in range(n) if score[t] == max(score, default=0) > 0]
    if not tied:
        return None
    k, _ = selector.pick([batches[live[0]].texts[t] for t in tied])
    return tied[k], tied


def _marginal(i: int, passing: list[int], tied: list[int]) -> float:
    """The probability that candidate `i` of a joke is the line picked: the sum over the tied threads t of
    P(thread t) times P(line | t), which is 1 when i is t's own passing candidate, 1 over the passing candidates when
    t's own was rejected and a line is borrowed, and 0 otherwise. This is what `pick_prob` logs, because replay and
    preference training weight a line by how likely it was to be played at all, not by how it happened to be reached.
    It assumes the selector is uniform (the frozen contract)."""
    own = 1.0 if i in tied else 0.0
    borrowed = sum(t not in passing for t in tied) / len(passing)
    return (own + borrowed) / len(tied)


async def _prepare(bit, mode, context, *, generator, fallback, safety, selector, tts, clock, log, jokes_per_bit, spawn):
    """Everything a bit needs before its first line: candidates, the safety verdicts and the first render.
    The bit plays one thread; a joke whose candidate in that thread was rejected takes another thread's passing
    candidate, and a joke with none falls back to the hand-checked bank (rules only)."""
    age = context["age"]
    batches: list[Batch | None] = []
    reasons: list[str] = []
    for joke in range(jokes_per_bit):
        try:
            batches.append(await asyncio.to_thread(generator.generate, mode, bit, joke, context))
            reasons.append("")
        except Exception as e:  # a bad reply or a dead provider must not stop the show
            batches.append(None)
            reasons.append(f"generation failed: {type(e).__name__}: {e}")
    if prefetch := getattr(safety, "prefetch", None):  # the LLM layer judges the whole bit in one call
        await asyncio.to_thread(prefetch, [t for b in batches if b for t in b.texts], age)
    judged = [[_judge(safety.check, t, age) for t in b.texts] if b else [] for b in batches]
    passing = [[i for i, item in enumerate(items) if item["verdict"] == "pass"] for items in judged]
    for joke, batch in enumerate(batches):
        if batch and not passing[joke]:
            reasons[joke] = "every candidate rejected"
    picked = _pick_thread(batches, passing, selector)
    if picked and (told := getattr(generator, "played_thread", None)):
        told(bit, picked[0])  # so the next bit is not told a premise the room never heard
    plans: list[_Plan] = []
    for joke, batch in enumerate(batches):
        items, ok = judged[joke], passing[joke]
        if not ok and not fallback:
            _log.warning("bit %s joke %s: %s; no fallback, skipping", bit, joke, reasons[joke])
            if not batch:  # nothing was generated, so the row still needs one item saying why (invariant 6)
                items = [{"text": "", "verdict": "reject", "reason": reasons[joke]}]
        plan = _Plan(items, [], batch.prompt_id if batch else f"{generator.generator_id}:failed",
                     batch.prompt if batch else "", generator.generator_id)
        if ok:
            first = picked[0] if picked[0] in ok else ok[selector.pick([batch.texts[i] for i in ok])[0]]
            plan.options = [(first, _marginal(first, ok, picked[1]))] + [
                (i, _marginal(i, ok, picked[1])) for i in ok if i != first]
        if fallback:
            plan.bank = _bank_consult(plan, bit, joke, mode, context, reasons[joke], fallback=fallback, safety=safety,
                                      selector=selector)
            if not plan.options:
                _log.warning("bit %s joke %s: %s; using %s", bit, joke, reasons[joke], fallback.generator_id)
                await plan.bank()
        plans.append(plan)
    _start_render(plans, 0, tts, clock, spawn)
    if first := next((p for p in plans if p.render), None):
        # The bit is ready only when its first line is audio, or the bridge ends into a render wait. A failed render
        # is handled when the line is spoken, so it is only waited for here.
        await asyncio.gather(first.render, return_exceptions=True)
    return plans


def _bank_consult(plan: _Plan, bit, joke, mode, context, why: str, *, fallback, safety, selector):
    async def consult() -> None:
        plan.bank = None  # once per joke
        batch = await asyncio.to_thread(fallback.generate, mode, bit, joke, context)
        # The bank is hand-checked, so only the rules run on it: no API call while the provider is failing.
        rules = getattr(safety, "check_rules", safety.check)
        start = len(plan.items)
        for text in batch.texts:
            item = _judge(rules, text, context["age"])
            item["reason"] = f"hand-checked bank: {why or 'voice failed'} ({fallback.generator_id}); rules: {item['reason']}"
            plan.items.append(item)
        ok = [start + i for i, item in enumerate(plan.items[start:]) if item["verdict"] == "pass"]
        if ok:
            first, prob = selector.pick([plan.items[i]["text"] for i in ok])
            plan.options = [(ok[first], prob)] + [(i, prob) for i in ok if i != ok[first]]
    return consult


def _start_render(plans, joke, tts, clock, spawn) -> None:
    """Render the next playable line from `joke` on, so its audio is ready when the current one ends."""
    plan = next((p for p in plans[joke:] if p.options), None)
    if plan and plan.render is None:
        plan.render = spawn(_render(tts, plan.items[plan.options[0][0]]["text"], clock))


async def _speak(plan: _Plan, tts, clock):
    """Render the joke's lines in order of preference until one has audio: (item index, pick_prob, wav, seconds).
    A line the voice cannot say (symbols only, an engine error) is skipped with the reason in its item, then the
    bank is tried, and None means nothing could be said. One bad line never stops the show."""
    while True:
        while plan.options:
            idx, prob = plan.options.pop(0)
            task, plan.render = plan.render, None
            try:
                wav, seconds = await (task or _render(tts, plan.items[idx]["text"], clock))
                return idx, prob, wav, seconds
            except Exception as e:
                _log.warning("a line could not be voiced: %s: %s", type(e).__name__, e)
                plan.items[idx]["reason"] = f"{plan.items[idx]['reason'] or ''}; voice failed: {type(e).__name__}: {e}"
        if plan.bank is None:
            return None
        await plan.bank()


async def _joke(bit, joke, plans, *, tts, player, source, clock, log, config, spawn, pauses):
    """One joke, end to end. Returns the level for the bit's reward, or None when nothing was played, the room did not
    answer, or a pause overlapped the playback (D-92: the label is still logged, replay drops it by the same rule)."""
    plan = plans[joke]
    _start_render(plans, joke, tts, clock, spawn)
    _start_render(plans, joke + 1, tts, clock, spawn)
    spoken = await _speak(plan, tts, clock)
    log.write("candidates", bit=bit, joke=joke, prompt_id=plan.prompt_id, prompt=plan.prompt,
              generator_id=plan.generator_id, items=plan.items)
    if spoken is None:
        return None  # unlabeled: no reaction row, so replay sees the same thing
    chosen, prob, wav, tts_seconds = spoken
    text = plan.items[chosen]["text"]
    await pauses.running()  # the show holds while paused, so this line's span holds no pause that began before it
    start = clock.now()
    await player.play(wav, text)
    end = clock.now()
    spoiled = overlaps(start, end, pauses.intervals())  # no await since `end`, so a later pause cannot be missed
    log.write("joke_played", bit=bit, joke=joke, chosen=chosen, pick_prob=prob, text=text,
              tts_seconds=tts_seconds, audio_seconds=wav_seconds(wav), audio_start_at=start, audio_end_at=end)
    level, key, latency = await source.reaction(bit, joke, config.timeout_s)
    log.write("reaction", bit=bit, joke=joke, source=source.source, level=level, key=key, latency_s=latency)
    before = clock.now()
    while True:
        hold = config.holds[level]
        fix = await source.hold(bit, joke, hold, clock)
        if fix is None:
            break
        level, key, latency = fix  # last press wins; the hold restarts at the new level's length
        log.write("reaction", bit=bit, joke=joke, source=source.source, level=level, key=key, latency_s=latency)
    log.write("gap", bit=bit, joke=joke, planned_hold_s=hold, actual_gap_s=clock.now() - before)
    return None if spoiled else level


# A bit prepares after the previous one ends, so the wait is generation (deadline 12 s) then the filter (25 s), about
# 37 s at worst; at about 5 s a line, 8 lines cover it. Past that the provider is down, not slow. The 10 bridge lines
# repeat in a long stall. tests/test_generate.py keeps this in step with app.generate's deadlines. Cached and filler
# lines share this cap.
MAX_BRIDGES = 8
# The show-open wait was 7.5 to 11.5 s over 4 Gemini shows and a played line lasts 6.3 s on average (both measured
# from data/sessions), so two lines cover the worst wait and three leave margin for a slow provider without filling
# half a minute with lines nobody is rating.
OPEN_CAP = 3
# Rendering is serial on the CPU voice, so lines beyond the show-open bit and the next few would only queue in front
# of the show's own first lines.
CACHED_RENDERS = 15


@dataclass(eq=False)
class _Cached:
    """A cached line on its way to the room: rendered in the background, `ready` once it has a wav or has failed."""
    text: str
    mode: str
    source: dict  # session, bit, joke, mode: where the line came from, for the `bridge` row
    wav: bytes | None = None
    seconds: float = 0.0
    used: bool = False
    ready: asyncio.Event | None = None


def _cached_pool(cached: dict[str, list[CachedBit]] | None, safety, context: dict) -> tuple[list[list[_Cached]], list[dict]]:
    """(the cached bits as lines to play, the account of them for the log). Each line passed its own show's filter, not
    this one's (teen rooms are stricter), so `screen` re-checks every line with the rules layer, once, before the
    show plays, and keeps a teen room to bits from teen shows."""
    if not cached:
        return [], []
    check = getattr(safety, "check_rules", safety.check)
    kept, report = screen(cached, check, context["age"], context["venue"])
    return [[_Cached(line.text, b.mode, {"session": b.session, "bit": b.bit, "joke": line.joke, "mode": b.mode},
                     ready=asyncio.Event()) for line in b.lines] for b in kept], report


def _round_robin(pool: list[list[_Cached]], skip: list[_Cached]) -> list[_Cached]:
    """The lines still to render, one mode at a time in turn (each mode's bits best first), so a next bit of any mode
    with a cache finds a rendered line instead of one mode's top bits taking every slot."""
    by_mode: dict[str, list[_Cached]] = {}
    for group in pool:
        by_mode.setdefault(group[0].mode, []).extend(e for e in group if e not in skip and not e.ready.is_set())
    out = []
    while any(by_mode.values()):
        for lines in by_mode.values():
            if lines:
                out.append(lines.pop(0))
    return out


async def _render_cached(entries: list[_Cached], tts, clock, after: asyncio.Task | None = None) -> None:
    """In order, so the first line to play is the first to be ready. A line that will not render is just skipped.
    `after` is the render task this one chains behind, so the voice, which is serial, never has two cached lines
    asking at once and a bit's own first line waits behind at most one."""
    if after is not None:
        await asyncio.wait({after})
    for e in entries:
        try:
            e.wav, e.seconds = await _render(tts, e.text, clock)
        except Exception:  # filler must never stop the show
            _log.warning("a cached line did not render", exc_info=True)
        e.ready.set()


async def _play_cached(e: _Cached, player, *, bit, clock, log, pauses) -> None:
    """A cached line is a `bridge` row marked cached, never a reaction window (replay must see no new labels)."""
    e.used = True  # before the await: a line is never said twice in a show
    await pauses.running()
    start = clock.now()
    await player.play(e.wav, e.text)
    log.write("bridge", bit=bit, text=e.text, tts_seconds=e.seconds, audio_start_at=start, audio_end_at=clock.now(),
              kind="cached", source=e.source)


async def _open_cached(prep: asyncio.Task, entries: list[_Cached], player, *, clock, log, pauses) -> None:
    """Fill the show-open wait with one cached bit's lines, in order, while bit 0 prepares. Waits for the next line's
    render only while bit 0 is still not ready, so a render never delays the bit itself."""
    for e in entries[:OPEN_CAP]:
        await pauses.running()
        if prep.done():
            return
        if not e.ready.is_set():
            waiter = asyncio.ensure_future(e.ready.wait())
            try:
                await asyncio.wait({prep, waiter}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                waiter.cancel()  # also on an abort, which cancels this await
            if prep.done():
                return
        if e.wav is not None:
            await _play_cached(e, player, bit=0, clock=clock, log=log, pauses=pauses)


def _next_cached(pool: list[list[_Cached]], mode: str | None) -> _Cached | None:
    """The best rendered line not yet said, from the next bit's mode when there is one."""
    ready = [e for group in pool for e in group if e.wav is not None and not e.used]
    return next((e for e in ready if e.mode == mode), ready[0] if ready else None)


async def _bridge(prep: asyncio.Task, rendered: list, player, used: list, *, bit, clock, log, pauses,
                  pool: list[list[_Cached]] = (), mode: str | None = None) -> None:
    """Fill the wait for the next bit with cached lines first (the next bit's mode preferred), then in-character
    bridge lines. Uses whatever has rendered so far and never blocks on the rest; a line is not repeated while an
    unused one remains. Each line played is a `bridge` row, because it is audio the room heard (invariant 6)."""
    for _ in range(MAX_BRIDGES):
        await pauses.running()  # also what keeps the filler path below free of an await before `start`
        if prep.done():
            return
        if cached := _next_cached(pool, mode):
            await _play_cached(cached, player, bit=bit, clock=clock, log=log, pauses=pauses)
            continue
        if not rendered:
            return
        unused = [b for b in rendered if b[0] not in used]
        text, wav, tts_seconds = unused[0] if unused else rendered[len(used) % len(rendered)]
        used.append(text)
        start = clock.now()
        await player.play(wav, text)
        log.write("bridge", bit=bit, text=text, tts_seconds=tts_seconds, audio_start_at=start, audio_end_at=clock.now(),
                  kind="filler")


async def run_show(director, room: str, context: dict, *, generator: Generator, safety: SafetyFilter,
                   selector: Selector, tts: TTS, player: Player, source: RewardSource, clock: Clock, log: SessionLog,
                   explore_seed: int | None = None, operator: str = "sim", git_sha: str = "unknown",
                   audio: dict | None = None, n_bits: int = 8, jokes_per_bit: int = 5, config=REACTION,
                   fallback: Generator | None = None, bridge_lines: list[str] | None = None,
                   cached: dict[str, list[CachedBit]] | None = None, pauses: Pauses | None = None) -> None:
    """One show. `director` is an EpsilonSoft over LinUCB, already warm: the caller replays the career into it
    (app.director.load_director) so this loop never decides which rooms count."""
    modes = AudienceEnv().modes
    pauses = Pauses() if pauses is None else pauses  # the server's handler writes the rows and updates this; a headless show never pauses
    inner = director.inner
    tasks: set[asyncio.Task] = set()  # the live ones, so an abort can cancel them and a long show does not keep every render

    def spawn(coro) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return task

    rendered: list[tuple[str, bytes, float]] = []  # bridge lines that are ready; a failed one is just not here
    bridges_used: list[str] = []

    async def render_bridges():
        for text in bridge_lines:
            try:
                wav, seconds = await _render(tts, text, clock)
                rendered.append((text, wav, seconds))
            except Exception:  # filler must never stop the show
                _log.warning("bridge line did not render: %r", text, exc_info=True)
    pool, cache_report = _cached_pool(cached, safety, context)
    log.write("show_start", room=room, context=context,
              policy={"name": "linucb", "epsilon": director.eps, "alpha": inner.alpha, "prior": "uniform",
                      "seed": explore_seed},
              generator_id=generator.generator_id, selector_id=selector.selector_id, git_sha=git_sha,
              operator=operator, audio=audio or {"consent": False, "path": None})
    if cache_report:  # invariant 6: what the show could play, and what the re-check dropped
        log.write("cache", bits=cache_report)
    opened: asyncio.Task | None = None
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
            prep = spawn(_prepare(bit, modes[action], context, generator=generator, fallback=fallback, safety=safety,
                                  selector=selector, tts=tts, clock=clock, log=log, jokes_per_bit=jokes_per_bit,
                                  spawn=spawn))
            if bit == 0 and pool:
                # Bit 0 cannot start for 7 to 12 s (generate, filter, first render): play the best cached bit of its
                # mode, else the best of any, and render it now, while the voice is free.
                opener = next((g for g in pool if g[0].mode == modes[action]), pool[0])
                opened = spawn(_render_cached(opener[:OPEN_CAP], tts, clock))
                await _open_cached(prep, opener, player, clock=clock, log=log, pauses=pauses)
            elif (bridge_lines or pool) and bit > 0:
                await _bridge(prep, rendered, player, bridges_used, bit=bit, clock=clock, log=log, pauses=pauses, pool=pool,
                              mode=modes[action])
            plans = await prep
            if bit == 0:
                # after bit 0's first line, so the voice is not busy when the show opens
                rest = _round_robin(pool, opener[:OPEN_CAP]) if pool else []
                if rest:
                    spawn(_render_cached(rest[:CACHED_RENDERS], tts, clock, after=opened))
                if bridge_lines:
                    spawn(render_bridges())
            levels = [await _joke(bit, joke, plans, tts=tts, player=player, source=source, clock=clock, log=log,
                                  config=config, spawn=spawn, pauses=pauses) for joke in range(jokes_per_bit)]
            reward = reward_v1(levels)
            log.write("bit_end", bit=bit, n_labels=sum(lv is not None for lv in levels), reward_version=VERSION,
                      reward=reward)
            if reward is not None:
                director.update(obs, action, reward)
            last_action, last_reward = action, reward
    except BaseException:  # includes cancellation: a closed tab must still leave a readable log
        log.write("show_end", status="aborted")
        raise
    finally:
        for task in list(tasks):
            task.cancel()  # a render nobody will play, or a bit nobody will hear
    log.write("show_end", status="finished")
