"""Exp 22: the Stage 0 acceptance run (docs/plans/live-loop.md, T11). Scores the shows the user ran.

Usage:
    python -m sim.acceptance <session id> [<session id> ...] [--dir data/sessions] [--record]

Reads the logs (and the recordings beside them) and reports, per show, whether it is a clean one, then the plan's
pass checks. Plain, it prints and saves nothing, so trying ids out never touches the evidence. `--record` is the only
way to write `reports/metrics/acceptance.json`, the Exp 22 evidence: numbers and the session ids only, no text, no
audio, no paths. It holds one run, the last one recorded.

Checks, from the plan:
  (a) 3 simulated shows in a row, each valid and finished (no abort). An unreadable log among the ids fails (a)
      outright: it has no start time to place it in the row, so pass only the shows meant to count;
  (b) at least 1 live show, valid and finished;
  (c) label spread: a live show where one key takes more than 60% of the presses is flagged. A judgment call in
      the plan, so it is reported and never blocks;
  (d) real per-bit reward noise against the sim's, and "rerun Exp 18" when it differs by more than half.

What "the sim's 0.15" is: `sim.policies.BIT_REWARD_SD`, also LinUCB's `alpha`. It is a rounded stand-in for the sd of
a bit's reward (mean of 5 levels / 3) that the sim produces from its latent noise (sd 1.0 in `audience.yaml`): measured
at the best mode, 0.128 to 0.140 over the 27 rooms, mean 0.136 (`evaluate_env.env_reward_sd`). It is the spread of
the 5 jokes' mean around a fixed expectation, so the live analogue is the spread of a bit's jokes around that bit's
own mean, divided down to the mean of 5: sqrt(pooled within-bit variance of level/3 x mean(1/n labels)).
Joke-to-joke differences in funniness inside a bit count as noise here, as the sim's per-joke draws do; differences
between bits do not (they are what the director is trying to learn).
"""
import argparse
import hashlib
import json
import math
import pathlib
import sys
import wave

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app.director import joke_levels  # noqa: E402
from app.log import KEY_LEVELS, SESSIONS, read_session  # noqa: E402
from app.pauses import intervals, paused_jokes  # noqa: E402
from app.reward import reward_v1  # noqa: E402
from sim import cache  # noqa: E402
from sim.env import MAX_LEVEL  # noqa: E402
from sim.policies import BIT_REWARD_SD  # noqa: E402

SPREAD_LIMIT = 0.60  # plan: one key over 60% of presses
SIM_SD_MEASURED = 0.136  # evaluate_env.env_reward_sd, mean over the 27 rooms (0.128 to 0.140)
RERUN_FACTOR = 0.5  # "differs by more than half"
SIMS_IN_A_ROW = 3
BANK_PREFIX = "bank"  # generator ids of BankGenerator, e.g. bank-v1
KEY_OF = {level: key for key, level in KEY_LEVELS.items()}


def label_spread(levels: list[int | None]) -> dict:
    """Count and share per key of each joke's final press (the last one), `none` for a window nobody answered. Fed
    from the raw reaction rows, so a joke dropped from the reward for a pause still counts as the press the operator
    made: this check is about the operator's key bias, not the reward. `presses` excludes `none`, and the flag looks at presses only."""
    counts = {key: 0 for key in (*KEY_LEVELS, "none")}
    for level in levels:
        counts["none" if level is None else KEY_OF[level]] += 1
    presses = sum(counts[k] for k in KEY_LEVELS)
    share = {k: counts[k] / len(levels) if levels else 0.0 for k in counts}
    top = max(KEY_LEVELS, key=lambda k: counts[k])
    return {"counts": counts, "share": share, "presses": presses, "top_key": top if presses else None,
            "top_share_of_presses": counts[top] / presses if presses else 0.0,
            "flagged": presses > 0 and counts[top] / presses > SPREAD_LIMIT}


def bit_levels(rows: list[dict], keep_paused: bool = False) -> list[list[int | None]]:
    """Per finished bit (one with a `bit_end` row), the final level of each joke; a joke whose playback a pause
    interrupted is None unless `keep_paused` (D-92)."""
    return [joke_levels(rows, r["bit"], keep_paused) for r in rows if r["event"] == "bit_end"]


def noise_stats(bits: list[list[int | None]]) -> dict:
    """Sufficient statistics of the estimate, which add across shows: bits that count as a reward (`reward_v1` is not
    None, so the rule follows the frozen reward), their pooled sum of squares around each bit's own mean, df = sum of
    (n - 1), and sum of 1/n."""
    ss = df = inv_n = 0.0
    n_bits = 0
    for levels in bits:
        if reward_v1(levels) is None:
            continue
        given = [lv / MAX_LEVEL for lv in levels if lv is not None]
        mean = sum(given) / len(given)
        ss += sum((x - mean) ** 2 for x in given)
        df += len(given) - 1
        inv_n += 1 / len(given)
        n_bits += 1
    return {"n_bits": n_bits, "df": int(df), "ss": ss, "inv_n": inv_n}


def pool(parts: list[dict]) -> dict:
    return {k: sum(p[k] for p in parts) for k in ("n_bits", "df", "ss", "inv_n")}


def finish_noise(stats: dict) -> dict:
    """The live analogue of the sim's per-bit reward sd (module docstring), from `noise_stats` of one show or of
    several pooled. `ci95` is a normal approximation on the pooled sd with df = sum(n - 1); it assumes independent,
    roughly normal residuals, which 3-4 level scores on few bits only roughly are, so read it as a guide."""
    if not stats["df"]:
        return {"n_bits": 0, "df": 0, "joke_sd": None, "bit_sd": None, "scale": None, "ci95": None}
    df = stats["df"]
    joke_sd = math.sqrt(stats["ss"] / df)
    scale = math.sqrt(stats["inv_n"] / stats["n_bits"])
    se = joke_sd / math.sqrt(2 * df)
    return {"n_bits": stats["n_bits"], "df": df, "joke_sd": joke_sd, "bit_sd": joke_sd * scale, "scale": scale,
            "ci95": [max(0.0, joke_sd - 1.96 * se) * scale, (joke_sd + 1.96 * se) * scale]}


def noise(bits: list[list[int | None]]) -> dict:
    return finish_noise(noise_stats(bits))


def noise_verdict(est: dict) -> dict:
    """Against the plan's 0.15: rerun Exp 18 when the estimate is more than half away from it. `decided` says whether
    the whole interval is on one side of the band's edge, so a rerun call on a few bits is marked as a guess."""
    if est["bit_sd"] is None:
        return {"rerun_exp18": None, "decided": False, "band": None}
    low, high = BIT_REWARD_SD * (1 - RERUN_FACTOR), BIT_REWARD_SD * (1 + RERUN_FACTOR)
    point = not low <= est["bit_sd"] <= high
    lo, hi = est["ci95"]
    inside, outside = low <= lo and hi <= high, hi < low or lo > high
    return {"rerun_exp18": point, "decided": inside or outside, "band": [low, high],
            "relative_to_sim": est["bit_sd"] / BIT_REWARD_SD - 1}


def wav_seconds(path: pathlib.Path) -> float | None:
    try:
        with wave.open(str(path)) as w:
            return w.getnframes() / w.getframerate()
    except (OSError, EOFError, wave.Error):
        return None


def bank_jokes(rows: list[dict]) -> int:
    """Jokes played from the bank. Two ways: a bank-only show (`GENERATOR=bank`) logs the bank's id on every
    `candidates` row, and in an llm show a fallback line's item reason starts with "hand-checked bank"
    (backend/app/show.py). A passing generated item has reason null."""
    cands = {(r["bit"], r["joke"]): r for r in rows if r["event"] == "candidates"}
    n = 0
    for r in rows:
        if r["event"] == "joke_played":
            c = cands[(r["bit"], r["joke"])]
            reason = c["items"][r["chosen"]]["reason"] or ""
            n += c["generator_id"].startswith(BANK_PREFIX) or reason.startswith("hand-checked bank")
    return n


def pause_report(rows: list[dict]) -> dict:
    """Pauses in one show (D-92): how many, how many seconds, and how many jokes lost their label to one. An unmatched
    pause (the show ended paused) counts to the last row's time. Seconds are on the show clock, and the log's `at` is
    the same timebase for a real show."""
    cut = next((i for i, r in enumerate(rows) if r["event"] == "show_end"), len(rows))
    rows = rows[:cut + 1]  # a row after show_end is not part of the show
    end = rows[-1]["at"]
    spans = intervals(rows)
    bits = {r["bit"] for r in rows if r["event"] == "joke_played"}
    return {"count": len(spans), "paused_s": sum(max(0.0, min(q, end) - p) for p, q in spans),
            "jokes_dropped": sum(len(paused_jokes(rows, b)) for b in bits)}


def audio_coverage(rows: list[dict], directory: pathlib.Path) -> dict | None:
    audio = rows[0]["audio"]
    if audio.get("path") is None:
        return None
    end = next((r["at"] for r in rows if r["event"] == "show_end"), rows[-1]["at"])
    span, seconds = end - audio["start_at"], wav_seconds(directory / audio["path"])
    # The browser keeps recording while paused (its capture path is not the playback context), so the file spans the
    # pauses too and the expected span is not reduced by them.
    return {"wav_s": seconds, "log_span_s": span, "coverage": None if seconds is None or span <= 0 else seconds / span}


def show_report(session: str, directory: pathlib.Path) -> dict:
    """One show's row. A log that does not read is reported as invalid, with the reason, and nothing else about it."""
    path = directory / f"{session}.jsonl"
    try:
        rows = read_session(path)
        if not rows or rows[0]["event"] != "show_start":
            raise ValueError("log does not begin with show_start")
    except (OSError, ValueError) as e:
        return {"session": session, "valid": False, "error": f"{type(e).__name__}: {e}"}
    start, ends = rows[0], [r for r in rows if r["event"] == "show_end"]
    played = [r for r in rows if r["event"] == "joke_played"]
    bits = bit_levels(rows)
    levels = [lv for bit in bit_levels(rows, keep_paused=True) for lv in bit]
    banked = bank_jokes(rows)
    stats = noise_stats(bits)
    ratings = [r["value"] for r in rows if r["event"] == "rating"]
    pauses = pause_report(rows)
    status = ends[-1]["status"] if ends else "none"
    return {"session": session, "valid": True, "error": None, "at": start["at"], "room": start["room"],
            "status": status, "generator_id": start["generator_id"],
            "n_jokes": len(played), "n_bits": sum(r["event"] == "bit_end" for r in rows),
            "bank_jokes": banked, "bank_share": banked / len(played) if played else None,
            "n_bridges": sum(r["event"] == "bridge" for r in rows), "labels": label_spread(levels),
            "rating": ratings[-1] if ratings else None, "pauses": pauses,
            "rating_counts": status == "finished" and pauses["count"] == 0, "audio": audio_coverage(rows, directory),
            "noise_stats": stats, "noise": finish_noise(stats)}


def clean(show: dict) -> bool:
    return show["valid"] and show["status"] == "finished"


def sims_in_a_row(shows: list[dict]) -> bool:
    """Three simulated shows, consecutive in start order among the sim shows given, all clean. Any unreadable log among
    the ids fails it: it has no start time, so it cannot be placed in the row, and quietly skipping it could hide a
    broken show between two good ones."""
    if not all(s["valid"] for s in shows):
        return False
    sims = [s for s in sorted((s for s in shows if s["valid"]), key=lambda s: s["at"]) if s["room"] == "sim"]
    return any(all(clean(s) for s in sims[i:i + SIMS_IN_A_ROW]) for i in range(len(sims) - SIMS_IN_A_ROW + 1))


def rating_summary(shows: list[dict]) -> dict:
    """The reported outcome (invariant 5): ratings of finished shows with no pause. A paused or aborted show's rating
    stays listed in `excluded` with the reason, because half a show is not comparable to a whole one (D-90)."""
    rated = [s for s in shows if s["valid"] and s["rating"] is not None]
    values = [s["rating"] for s in rated if s["rating_counts"]]
    return {"n": len(values), "mean": sum(values) / len(values) if values else None, "values": values,
            "excluded": [{"session": s["session"], "rating": s["rating"],
                          "reason": "aborted" if s["status"] != "finished" else "paused"}
                         for s in rated if not s["rating_counts"]]}


def pooled_noise(shows: list[dict]) -> dict:
    """Over every finished live show given: the pooled estimate and its verdict."""
    live = [s for s in shows if s["valid"] and s["room"] == "live" and s["status"] == "finished"]
    est = finish_noise(pool([s["noise_stats"] for s in live]))
    return {"n_live_shows": len(live), **est, "sim_sd_plan": BIT_REWARD_SD, "sim_sd_measured": SIM_SD_MEASURED,
            **noise_verdict(est)}


def run(sessions: list[str], key: str = "", directory: str | pathlib.Path = SESSIONS) -> dict:
    """`key` only distinguishes cache entries (see `digest`); the logs are read from `directory`."""
    directory = pathlib.Path(directory)
    shows = [show_report(s, directory) for s in sessions]
    live = [s for s in shows if s["valid"] and s["room"] == "live"]
    return {"sessions": list(sessions), "shows": shows,
            "pass_a_sims_in_a_row": sims_in_a_row(shows),
            "pass_b_live_show": any(clean(s) for s in live),
            "flag_c_label_spread": [s["session"] for s in live if s["labels"]["flagged"]],
            "ratings": rating_summary(shows), "noise": pooled_noise(shows)}


def digest(sessions: list[str], directory: str | pathlib.Path = SESSIONS) -> str:
    """The cache key's stand-in for the files the report reads: each log's bytes and, for a recorded show, its WAV's
    size and modification time, so an edited log, a newly rated one or a replaced recording recomputes."""
    directory, h = pathlib.Path(directory), hashlib.sha256()
    for sid in sessions:
        path = directory / f"{sid}.jsonl"
        h.update(path.read_bytes() if path.exists() else b"missing")
        try:
            audio = read_session(path)[0]["audio"]
            wav = directory / audio["path"] if audio.get("path") else None
        except (OSError, ValueError, KeyError, IndexError):
            wav = None
        if wav is not None:
            stat = wav.stat() if wav.exists() else None
            h.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode() if stat else b"nowav")
    return h.hexdigest()[:16]


def record(sessions: list[str], directory: str | pathlib.Path = SESSIONS) -> dict:
    """Score the shows and save the result as the Exp 22 evidence, replacing the previous record. The only writer."""
    return cache.run_or_load("acceptance", lambda sessions, key: run(sessions, key, directory), rerun=True,
                             sessions=sessions, key=digest(sessions, directory))


def stored() -> dict | None:
    """The last result `record` saved, without needing the logs (they stay on the user's machine); None before a run."""
    path = cache.METRICS / "acceptance.json"
    return json.loads(path.read_text(encoding="utf-8"))["results"] if path.exists() else None


def pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def table(res: dict) -> str:
    lines = ["| Show | Room | Valid | Status | Generator | Jokes | Bank | Bridges | W / K / Space / D / none | Pauses (s, jokes dropped) | Rating | Audio |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in res["shows"]:
        if not s["valid"]:
            lines.append(f"| {s['session']} | - | NO: {s['error']} | - | - | - | - | - | - | - | - | - |")
            continue
        c = s["labels"]["counts"]
        cov = (s["audio"] or {}).get("coverage")
        p = s.get("pauses", {"count": 0})  # a record saved before D-92 has no pause fields
        paused = "-" if not p["count"] else f"{p['count']} ({p['paused_s']:.0f} s, {p['jokes_dropped']})"
        rating = "-" if s["rating"] is None else f"{s['rating']}" + ("" if s.get("rating_counts", True) else " (not counted)")
        lines.append(f"| {s['session']} | {s['room']} | yes | {s['status']} | {s['generator_id']} | {s['n_jokes']} | "
                     f"{pct(s['bank_share'])} | {s['n_bridges']} | "
                     f"{c['W']} / {c['K']} / {c['Space']} / {c['D']} / {c['none']} | {paused} | {rating} | "
                     f"{'not recorded' if s['audio'] is None else pct(cov)} |")
    return "\n".join(lines)


def verdict(res: dict) -> str:
    n = res["noise"]
    out = [f"(a) {SIMS_IN_A_ROW} simulated shows in a row, all clean: {'PASS' if res['pass_a_sims_in_a_row'] else 'NOT MET'}",
           f"(b) a live show, valid and finished: {'PASS' if res['pass_b_live_show'] else 'NOT MET'}",
           "(c) label spread, one key over 60% of presses (reported, never blocks): "
           + (", ".join(res["flag_c_label_spread"]) + " FLAGGED" if res["flag_c_label_spread"] else "none flagged")]
    if r := res.get("ratings"):
        out.append("reported rating: " + (f"mean {r['mean']:.2f} of {r['n']} show(s)" if r["n"] else "no counted rating yet")
                   + "".join(f"; {x['session']} not counted ({x['reason']})" for x in r["excluded"]))
    if n.get("bit_sd") is None:
        out.append("(d) per-bit reward noise: no live bit with 3 or more labels yet")
    else:
        lo, hi = n["ci95"]
        out.append(f"(d) per-bit reward noise {n['bit_sd']:.3f} (95% {lo:.3f} to {hi:.3f}) from {n['n_bits']} bits of "
                   f"{n['n_live_shows']} live show(s), against the sim's {n['sim_sd_plan']} (measured {n['sim_sd_measured']}); "
                   f"band {n['band'][0]:.3f} to {n['band'][1]:.3f}: "
                   + ("RERUN Exp 18" if n["rerun_exp18"] else "no rerun needed")
                   + ("" if n["decided"] else " (the interval crosses the band's edge: a guess, not a finding)"))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sessions", nargs="+", help="session ids: the file names in data/sessions without .jsonl")
    ap.add_argument("--dir", default=str(SESSIONS))
    ap.add_argument("--record", action="store_true", help="save the result as the Exp 22 evidence (replaces the last)")
    a = ap.parse_args()
    res = record(a.sessions, a.dir) if a.record else run(a.sessions, directory=a.dir)
    print(table(res), "", verdict(res), sep=chr(10))
    print("recorded to reports/metrics/acceptance.json" if a.record else "not recorded (add --record to keep this as the Exp 22 evidence)")


if __name__ == "__main__":
    main()
