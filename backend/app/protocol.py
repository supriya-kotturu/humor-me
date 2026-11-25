"""Client-to-server websocket messages, validated before anything else sees them (backend/README.md lists both ways)."""
import json

from app.log import KEY_LEVELS
from sim.env import AGES, ENERGIES, VENUES


class ProtocolError(ValueError):
    pass


def _int(msg: dict, field: str) -> int:
    value = msg.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProtocolError(f"{field} must be a non-negative integer")
    return value


def check_pause(kind: str, *, running: bool, paused: bool) -> None:
    """Order rules for `pause` and `resume`, given the session's state. Raises ProtocolError; the show goes on."""
    if not running:
        raise ProtocolError(f"{kind} refused: no show is running")
    if kind == "pause" and paused:
        raise ProtocolError("pause refused: the show is already paused")
    if kind == "resume" and not paused:
        raise ProtocolError("resume refused: the show is not paused")


def parse(frame: dict) -> dict:
    """The ASGI websocket frame in, a checked message out. Raises ProtocolError; the connection stays open."""
    text = frame.get("text")
    if text is None:
        raise ProtocolError("expected a text frame")
    try:
        msg = json.loads(text)
    except ValueError:
        raise ProtocolError("not valid JSON") from None
    if not isinstance(msg, dict):
        raise ProtocolError("message must be a JSON object")
    kind = msg.get("type")
    if kind == "played":
        return {"type": kind}
    if kind in ("pause", "resume"):  # order is checked by `check_pause`: it needs the show's state
        return {"type": kind}
    if kind == "audio_health":
        health = {}
        for field in ("produced_s", "elapsed_s", "gap_ms", "wall_s", "rate"):
            value = msg.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1e9:
                raise ProtocolError(f"{field} must be a non-negative number")
            health[field] = value
        return {"type": kind, **health}
    if kind == "audio_start":  # when the first sample was captured, by the capture clock (epoch ms)
        wall_ms = msg.get("wall_ms")
        if isinstance(wall_ms, bool) or not isinstance(wall_ms, (int, float)) or not 0 < wall_ms < 1e14:
            raise ProtocolError("wall_ms must be a positive number")
        return {"type": kind, "wall_ms": wall_ms}
    if kind == "audio_failed":
        reason = msg.get("reason")
        return {"type": kind, "reason": reason[:200] if isinstance(reason, str) and reason.strip() else None}
    if kind == "reaction":
        key = msg.get("key")
        if not isinstance(key, str) or key not in KEY_LEVELS:
            raise ProtocolError(f"key must be one of {tuple(KEY_LEVELS)}")
        return {"type": kind, "bit": _int(msg, "bit"), "joke": _int(msg, "joke"), "key": key}
    if kind == "rating":
        value = msg.get("value")
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 5:
            raise ProtocolError("value must be an integer from 1 to 5")
        return {"type": kind, "value": value}
    if kind == "start":
        context = msg.get("context")
        if not isinstance(context, dict):
            raise ProtocolError("context must be an object")
        for field, allowed in (("venue", VENUES), ("energy", ENERGIES), ("age", AGES)):
            if context.get(field) not in allowed:
                raise ProtocolError(f"context.{field} must be one of {allowed}")
        if msg.get("room") not in ("sim", "live"):
            raise ProtocolError("room must be 'sim' or 'live'")
        operator = msg.get("operator")
        if not isinstance(operator, str) or not operator.strip():
            raise ProtocolError("operator must be a non-empty string")
        consent = msg.get("consent", False)
        if not isinstance(consent, bool):
            raise ProtocolError("consent must be true or false")
        return {"type": kind, "context": {f: context[f] for f in ("venue", "energy", "age")}, "room": msg["room"],
                "operator": operator, "consent": consent}
    raise ProtocolError(f"unknown message type {kind!r}")
