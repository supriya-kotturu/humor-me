"""Exp 23: do cached bits fill the show-open silence? (docs/agent/decisions-detail.md, D-91)

Usage:
    python -m sim.evaluate_cached <session id> [<session id> ...] [--dir data/sessions] [--record]

Scores shows the user ran (3 simulated, `GENERATOR=llm`, after at least one live show gave the cache). Plain, it prints
and saves nothing; `--record` writes `reports/metrics/cached_bits.json`: numbers and session ids only, no text.

Bar (fixed before any run, changed once after review and before any run): on 3 simulated shows with GENERATOR=llm
whose logged cache (the `cache` row) held a bit of bit 0's own mode with a line that passed the re-check at the start,
the show-open silence (show_start to the first audio of any kind, cached or joke) is at most 2 s on every one. A show
qualifies by what it had at the start, not by what it played: a slow or failed cached render is a MISS, not an
exclusion. A show with no such cached bit is listed and does not count toward the 3. The between-bit silence is
reported beside it, with no bar: this change does not target it, and Exp 20's bar (no silence over 2 s) judges it.
"""
import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app.log import SESSIONS, read_session  # noqa: E402
from sim import cache  # noqa: E402
from sim.evaluate_live import silences  # noqa: E402

BAR_OPEN_S = 2.0
SHOWS_NEEDED = 3
BANK_PREFIX = "bank"
AUDIO = ("joke_played", "bridge")


def open_silence(rows: list[dict]) -> float | None:
    """Seconds from `show_start` to the first audio of any kind (a joke or a bridge row, cached or filler); None when
    the show played nothing. Never negative: a coarse clock can stamp the audio before the row."""
    start = next((r["at"] for r in rows if r["event"] == "show_start"), None)
    first = min((r["audio_start_at"] for r in rows if r["event"] in AUDIO), default=None)
    return None if start is None or first is None else max(0.0, first - start)


def between_bit_silence(rows: list[dict]) -> list[float]:
    """Dead air at every bit boundary after the show's first audio, from the previous audio's end or its hold's end
    (the same measure as Exp 20, so the two compare). A cached line or a filler line counts as audio."""
    return silences(rows)["between"]


def had_cache_for_bit_zero(rows: list[dict]) -> bool:
    """True when the show's `cache` row held a bit of the mode bit 0 played with at least one line that passed the
    re-check (the bar's precondition, decided from what the show had at its start)."""
    mode = next((r["action"] for r in rows if r["event"] == "bit_decision" and r["bit"] == 0), None)
    return any(b["mode"] == mode and any(line["verdict"] == "pass" for line in b["lines"])
               for r in rows if r["event"] == "cache" for b in r["bits"])


def show_report(session: str, directory: pathlib.Path) -> dict:
    try:
        rows = read_session(directory / f"{session}.jsonl")
        if not rows or rows[0]["event"] != "show_start":
            raise ValueError("log does not begin with show_start")
    except (OSError, ValueError) as e:
        return {"session": session, "valid": False, "error": f"{type(e).__name__}: {e}"}
    start = rows[0]
    between = between_bit_silence(rows)
    bridges = [r for r in rows if r["event"] == "bridge"]
    ends = [r for r in rows if r["event"] == "show_end"]
    llm = not start["generator_id"].startswith(BANK_PREFIX)
    return {"session": session, "valid": True, "error": None, "at": start["at"], "room": start["room"],
            "status": ends[-1]["status"] if ends else "none", "generator_id": start["generator_id"],
            "counts": llm and start["room"] == "sim" and bool(ends) and ends[-1]["status"] == "finished"
                      and had_cache_for_bit_zero(rows),
            "open_silence_s": open_silence(rows),
            "n_bridges": len(bridges), "n_cached": sum(r.get("kind") == "cached" for r in bridges),
            "between_n": len(between), "between_max_s": max(between, default=None),
            "between_mean_s": sum(between) / len(between) if between else None}


def run(sessions: list[str], key: str = "", directory: str | pathlib.Path = SESSIONS) -> dict:
    """`key` only distinguishes cache entries (see `digest`); the logs are read from `directory`."""
    directory = pathlib.Path(directory)
    shows = [show_report(s, directory) for s in sessions]
    counted = [s for s in shows if s["valid"] and s["counts"]]
    opens = [s["open_silence_s"] for s in counted]
    betweens = [s["between_max_s"] for s in counted if s["between_max_s"] is not None]
    return {"sessions": list(sessions), "shows": shows, "bar_open_s": BAR_OPEN_S, "shows_needed": SHOWS_NEEDED,
            "n_counted": len(counted), "open_max_s": max(opens, default=None),
            "between_max_s": max(betweens, default=None),
            "pass_open": len(counted) >= SHOWS_NEEDED and all(x <= BAR_OPEN_S for x in opens)}


def digest(sessions: list[str], directory: str | pathlib.Path = SESSIONS) -> str:
    """The cache key's stand-in for the files read, so an edited log recomputes."""
    h = hashlib.sha256()
    for sid in sessions:
        path = pathlib.Path(directory) / f"{sid}.jsonl"
        h.update(path.read_bytes() if path.exists() else b"missing")
    return h.hexdigest()[:16]


def record(sessions: list[str], directory: str | pathlib.Path = SESSIONS) -> dict:
    return cache.run_or_load("cached_bits", lambda sessions, key: run(sessions, key, directory), rerun=True,
                             sessions=sessions, key=digest(sessions, directory))


def stored() -> dict | None:
    path = cache.METRICS / "cached_bits.json"
    return json.loads(path.read_text(encoding="utf-8"))["results"] if path.exists() else None


def secs(x: float | None) -> str:
    return "-" if x is None else f"{x:.1f} s"


def table(res: dict) -> str:
    lines = ["| Show | Room | Status | Counts toward the bar | Show-open silence | Cached lines | Between-bit max | Between-bit mean |",
             "|---|---|---|---|---|---|---|---|"]
    for s in res["shows"]:
        if not s["valid"]:
            lines.append(f"| {s['session']} | - | NO: {s['error']} | - | - | - | - | - |")
            continue
        lines.append(f"| {s['session']} | {s['room']} | {s['status']} | {'yes' if s['counts'] else 'no'} | "
                     f"{secs(s['open_silence_s'])} | {s['n_cached']} of {s['n_bridges']} bridges | "
                     f"{secs(s['between_max_s'])} | {secs(s['between_mean_s'])} |")
    return "\n".join(lines)


def verdict(res: dict) -> str:
    got = f"{res['n_counted']} of {res['shows_needed']} qualifying shows"
    if res["n_counted"] < res["shows_needed"]:
        return (f"Show-open silence at most {res['bar_open_s']:.0f} s: NOT DECIDED, only {got} (a qualifying show is a "
                f"finished simulated llm show whose cache held a bit of bit 0's mode)")
    return (f"Show-open silence at most {res['bar_open_s']:.0f} s on every one of {got}: "
            f"{'PASS' if res['pass_open'] else 'MISS'} (worst {secs(res['open_max_s'])}); "
            f"worst between-bit silence {secs(res['between_max_s'])}, no bar")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--dir", default=str(SESSIONS))
    ap.add_argument("--record", action="store_true", help="save as the Exp 23 evidence (replaces the last)")
    a = ap.parse_args()
    res = record(a.sessions, a.dir) if a.record else run(a.sessions, directory=a.dir)
    print(table(res), "", verdict(res), sep="\n")
    print("recorded to reports/metrics/cached_bits.json" if a.record else "not recorded (add --record)")


if __name__ == "__main__":
    main()
