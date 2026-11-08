"""Bit reward from operator labels (docs/plans/live-loop.md, "Reward"). Labels are the record; this is derived."""
from sim.env import MAX_LEVEL

VERSION = 1
MIN_LABELS = 3  # of a bit's 5 jokes; fewer and the room is too thinly read to teach the director


def reward_v1(levels) -> float | None:
    """Mean of the non-null levels / 3, or None with fewer than MIN_LABELS labels.
    Same arithmetic as AudienceEnv.step, so a sim run and a replay of its labels agree bit for bit."""
    given = [lv for lv in levels if lv is not None]
    if len(given) < MIN_LABELS:
        return None
    return sum(given) / len(given) / MAX_LEVEL
