"""Cached bits: the best-received bits of earlier live shows, kept as filler for the next show's silences
(docs/agent/decisions-detail.md, D-91). A cached line is never evaluated: no reaction window, no reward, no director
update. It was judged for its own show's audience, so `screen` re-checks it for this one before it can be said."""
from dataclasses import dataclass

from app.director import joke_levels
from app.reward import reward_v1
from app.safety import STANDARD_AGES


@dataclass(frozen=True)
class CachedLine:
    joke: int  # the line's index in its bit, for the log's `source`
    text: str


@dataclass(frozen=True)
class CachedBit:
    session: str
    bit: int
    mode: str
    reward: float  # recomputed from the reaction rows, never read from `bit_end.reward`
    lines: tuple[CachedLine, ...]
    age: str = "mixed"  # the source show's audience: the LLM teen rule never re-runs on a cached line
    venue: str = ""


def _from_bank(rows: list[dict], bit: int, joke: int, chosen: int) -> bool:
    """True when the line played was the hand-written bank's: the bank can say it again as a joke, so it is no filler."""
    cand = next((r for r in rows if r["event"] == "candidates" and (r["bit"], r["joke"]) == (bit, joke)), None)
    if cand is None:
        return False
    reason = (cand["items"][chosen]["reason"] or "") if 0 <= chosen < len(cand["items"]) else ""
    return cand["generator_id"].startswith("bank") or reason.startswith("hand-checked bank")


def popular_bits(sessions: list[list[dict]], per_mode: int = 2, min_labels: int = 3) -> dict[str, list[CachedBit]]:
    """The top `per_mode` bits of each mode, from live shows only: a simulated reward comes from made-up
    `audience.yaml` values, so it cannot say what was popular. Ranked by reward, ties to the later show. A bit with
    fewer than `min_labels` labels, no reward or no usable line is skipped, and a bit whose lines repeat a better
    one's is dropped. A line is not usable when the bank wrote it or the room scored it level 0."""
    found: list[tuple[float, float, int, CachedBit]] = []
    for rows in sessions:
        start = next((r for r in rows if r["event"] == "show_start"), None)
        if start is None or start["room"] != "live":
            continue
        actions = {r["bit"]: r["action"] for r in rows if r["event"] == "bit_decision"}
        for end in (r for r in rows if r["event"] == "bit_end"):
            bit = end["bit"]
            levels = joke_levels(rows, bit)
            reward = reward_v1(levels)
            if bit not in actions or reward is None or sum(lv is not None for lv in levels) < min_labels:
                continue
            # joke_levels has one level per joke that has a reaction row, in joke order
            level_of = dict(zip(sorted({r["joke"] for r in rows if r["event"] == "reaction" and r["bit"] == bit}), levels))
            lines = tuple(CachedLine(r["joke"], r["text"]) for r in rows
                          if r["event"] == "joke_played" and r["bit"] == bit and level_of.get(r["joke"]) != 0
                          and not _from_bank(rows, bit, r["joke"], r["chosen"]))
            if not lines:
                continue
            context = start["context"]
            found.append((reward, start["at"], bit, CachedBit(start["session"], bit, actions[bit], reward, lines,
                                                              context["age"], context["venue"])))
    found.sort(key=lambda f: f[:3], reverse=True)
    out: dict[str, list[CachedBit]] = {}
    seen: set[tuple[str, ...]] = set()
    for *_, cb in found:
        texts = tuple(line.text for line in cb.lines)
        if texts in seen or len(out.setdefault(cb.mode, [])) >= per_mode:
            continue
        seen.add(texts)
        out[cb.mode].append(cb)
    return {mode: bits for mode, bits in out.items() if bits}


def screen(cached: dict[str, list[CachedBit]], check, age: str, venue: str | None = None
           ) -> tuple[list[CachedBit], list[dict]]:
    """(bits to play, the log's account of every bit and line). Bits come with only the lines that pass, best reward
    first (a bit from this show's venue ahead of the rest). Three things drop a line, each with its reason in the account:
    - a teen room (an age outside STANDARD_AGES) takes only bits from shows of that same age class, because the LLM
      teen rule does not re-run on a cached line, so the source audience has to have been held to it;
    - `check` (the rules layer, no API call) fails it for this age, and a filter that cannot decide fails it;
    - the same text is already in the pool (a line is never said twice in a show)."""
    teen_room = age not in STANDARD_AGES
    ordered = sorted((b for bits in cached.values() for b in bits), key=lambda b: (b.venue != venue, -b.reward))
    kept, report, seen = [], [], set()
    for b in ordered:
        wrong_audience = teen_room and b.age in STANDARD_AGES
        lines = []
        for line in b.lines:
            if wrong_audience:
                verdict, reason = "reject", f"teen room, source show was {b.age}"
            elif line.text in seen:
                verdict, reason = "reject", "duplicate of a pooled line"
            else:
                try:
                    ok = check(line.text, age)
                    verdict, reason = ("pass" if ok.ok else "reject"), f"rules: {ok.reason}"
                except Exception as e:  # a filter that cannot decide must not let the line through
                    verdict, reason = "reject", f"filter error: {type(e).__name__}"
            if verdict == "pass":
                seen.add(line.text)
            lines.append({"joke": line.joke, "text": line.text, "verdict": verdict, "reason": reason})
        report.append({"session": b.session, "bit": b.bit, "mode": b.mode, "reward": b.reward, "source_age": b.age,
                       "lines": lines})
        passing = tuple(line for line, item in zip(b.lines, lines) if item["verdict"] == "pass")
        if passing:
            kept.append(CachedBit(b.session, b.bit, b.mode, b.reward, passing, b.age, b.venue))
    return kept, report


def cached_for(bridge_lines, sessions: list[list[dict]]) -> dict[str, list[CachedBit]] | None:
    """Cached bits only for the llm stack (the one with bridge lines). The bank stack has no filler today, so a cached
    line before every bit would be new audio there and would queue its voice in front of bit 0's first render."""
    return popular_bits(sessions) if bridge_lines else None
