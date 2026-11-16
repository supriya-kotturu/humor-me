"""The LLM generator: one call per bit returns K threads of the bit (thread-major in the reply, joke-major in a Batch).
Candidate i of every joke belongs to thread i, and the show plays one thread through (app.show._pick_thread).

Failure is the generator's to report and the show's to absorb: `generate` raises when the call or the reply is bad
(the same exception for every joke of that bit, with no second call), and the show falls back to the hand-checked
bank (app.show._prepare). Retries inside the call live in app.llm.
"""
import json

from app import llm, safety
from app.bank import BankGenerator
from app.show import Batch

K = 3  # threads per bit: enough for the filter to reject some and the selector to still choose
JOKES = 5
TEMPERATURE = 0.9
PROMPT_ID = "gen-v3"

# One identity, six attitudes (HumorGen personas as modes, CLAUDE.md invariant 3).
MODES = {
    "neurotic": "anxious and over-analytical: spirals about small things, rehearses worst cases",
    "cynic": "dry and deadpan: unimpressed, resigned, finds the system faintly absurd",
    "observer": "an observational comic: notices the small odd detail of everyday situations and says it plainly",
    "wordsmith": "built on wordplay: puns, double meanings, a phrase taken too literally",
    "optimist": "relentlessly cheerful about things going badly, which is the joke",
    "absurdist": "starts from a normal situation and commits to an escalating, illogical idea",
}
SYSTEM = """You write material for a live stand-up show that one character performs out loud: a self-aware, slightly anxious robot. It is always the same robot; the mode is only the attitude it takes in this bit.

Mode: {mode}: {mode_note}.

Write one bit in {k} alternative threads. A thread has its own premise angle and {jokes} jokes played in order (opener, tag, tag, tag, closer): the opener sets up the situation, each tag builds on it, and the closer calls back to a word or image from that same thread's opener. The {k} threads must take clearly different angles. Return each thread as its own list of {jokes} lines, in playing order. The show plays one thread through, or swaps single lines between threads.

Rules:
- Each line is one or two short sentences made for speaking aloud, under 30 words, with no stage directions or emoji.
- Aim every joke at the robot itself, technology, the situation, or things people say. Never at who anyone is: no jokes about bodies or appearance, no protected groups, no real or named people, nobody in the room singled out.
{teens}
Reply with JSON only, exactly this shape: {{"premises": ["premise of thread 0", ...], "threads": [[opener, tag, tag, tag, closer], ...]}} with {k} threads of {jokes} lines each."""
TEENS_RULE = "- The audience is teenagers: no alcohol, drugs, sexual content, innuendo or swearing."

# In character, hand-written, one breath each: they only have to fill a few seconds while a bit loads.
BRIDGES = (
    "Give me a second. I'm checking my notes, and they just say don't panic, which is not a note.",
    "Hang on, I'm reorganizing my thoughts. Most of them weren't where I left them.",
    "One moment please. I'm running a confidence check, and it's buffering, like everything else I own.",
    "Okay, next bit. Let me make sure I haven't ruined it in advance. I usually have.",
    "Bear with me. I'm rehearsing this part one more time, mostly to be nervous about it properly.",
    "Quick pause while I load the next bit. Please enjoy this brief moment of me worrying.",
    "Just a second. My second half is in another folder, and I'm afraid to open it.",
    "While I think of something, let's all agree this is going great. I'll go first.",
    "Sorry, I'm recalculating. My plan said everything would go well, so something is wrong with the plan.",
    "Almost there. I'm double-checking the next joke for problems, and so far the main problem is me.",
)

# Seconds: (per-attempt timeout, deadline over all attempts), and the retries after the first attempt.
GENERATOR_TIMING = (10.0, 12.0)
RETRIES = 2
# The filter's budget is sized by what a stall costs, not by how long a healthy call takes (1.3-2 s). A bit's
# preparation starts when the previous bit's last joke ends (show.run_show awaits every joke before the next
# iteration, because the next mode needs this bit's reward), so a longer wait is NOT hidden by playback: the bridge
# lines are the only cover, MAX_BRIDGES of them at about 5 s each. Exp 22: a 6 s attempt / 8 s deadline timed out
# 73 times in one show and sent 75% of its jokes to the bank. A fresh attempt at 12 s with 3 retries and a 25 s
# deadline lets a second try land after a first stall; failing closed past that is unchanged.
FILTER_TIMING = (12.0, 25.0)
FILTER_RETRIES = 3


class GenerationError(Exception):
    pass


def parse(reply: str) -> tuple[list[str], list[list[str]]]:
    """Accepts one shape only: {"premises": [...], "threads": [[JOKES lines] x K]}, thread-major. Returns the premises
    (one per thread when given) and the lines joke-major (lines[j][i] is joke j of thread i), which is what the show's
    batches hold. Any other shape or count raises, and so does a line with no letters (emoji or symbols only, which
    the voice cannot say): a thread cannot be repaired by guessing."""
    try:
        data = json.loads(llm.strip_fence(reply))
        threads = data["threads"]
        premises = data.get("premises") or []
        premises = [str(x) for x in premises] if isinstance(premises, list) else [str(premises)]
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise GenerationError(f"malformed reply: {type(e).__name__}") from e
    if (not isinstance(threads, list) or len(threads) != K
            or not all(isinstance(t, list) and len(t) == JOKES for t in threads)
            or not all(isinstance(line, str) and any(c.isalpha() for c in line) for t in threads for line in t)):
        shape = [len(t) if isinstance(t, list) else type(t).__name__ for t in threads] if isinstance(threads, list) else "?"
        raise GenerationError(f"expected {K} threads of {JOKES} lines with letters in them, got {shape}")
    return premises, [[threads[i][j].strip() for i in range(K)] for j in range(JOKES)]


class GeminiGenerator:
    """One instance per show. It tells the model which premises the room has heard, so a bit does not repeat one:
    only the thread the show actually played counts (`played_thread`), not the alternatives that were generated."""

    def __init__(self, client: llm.Client):
        self.client = client
        self.generator_id = f"{PROMPT_ID}:{client.model}"
        self._bits: dict[int, tuple[str, list[list[str]], list[str]] | Exception] = {}
        self._told: list[str] = []

    def generate(self, mode: str, bit: int, joke: int, context: dict) -> Batch:
        if bit not in self._bits:
            self._bits[bit] = self._fetch(mode, context)
        got = self._bits[bit]
        if isinstance(got, Exception):
            raise got
        prompt, lines, _ = got
        return Batch(f"{PROMPT_ID}:{mode}:{bit}:{joke}", prompt, lines[joke])

    def played_thread(self, bit: int, thread: int) -> None:
        got = self._bits.get(bit)
        if not isinstance(got, Exception) and got and thread < len(got[2]):
            self._told.append(got[2][thread])

    def _fetch(self, mode: str, context: dict) -> tuple[str, list[list[str]], list[str]] | Exception:
        teens = TEENS_RULE if context["age"] not in safety.STANDARD_AGES else ""
        system = SYSTEM.format(mode=mode, mode_note=MODES[mode], k=K, jokes=JOKES, teens=teens)
        user = json.dumps({"venue": context["venue"], "energy": context["energy"], "premises_already_told": self._told})
        try:
            premises, lines = parse(self.client.chat(system, user, json_reply=True, temperature=TEMPERATURE))
        except (llm.LLMError, GenerationError) as e:
            return e
        return f"{system}\n\n{user}", lines, premises


class LiveSafety:
    """The show's filter: rules, then one LLM verdict call per bit (`prefetch`), then cached `check`s."""

    def __init__(self, client: llm.Client):
        self.client = client

    def prefetch(self, texts: list[str], age: str) -> None:
        safety.prefetch(texts, age, self.client)

    def check(self, text: str, age: str) -> safety.Verdict:
        return safety.check(text, age)

    def check_rules(self, text: str, age: str) -> safety.Verdict:
        """The rules alone, for the bank fallback: hand-checked lines need no API call."""
        return safety.check_rules(text, age)


def llm_stack(room: str):
    """(generator, safety, fallback, bridge lines) for a show on the real model. Raises SafetyNotReady for a live
    room without the slur list. The deadlines keep a failing provider inside the bridge lines' cover."""
    safety.require_ready(room)
    return (GeminiGenerator(llm.from_env(GENERATOR_TIMING[0], retries=RETRIES, deadline_s=GENERATOR_TIMING[1])),
            LiveSafety(llm.from_env(FILTER_TIMING[0], retries=FILTER_RETRIES, deadline_s=FILTER_TIMING[1])),
            BankGenerator(), list(BRIDGES))
