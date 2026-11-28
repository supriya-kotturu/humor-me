"""The LLM generator (live prompt gen-v5 since D-111, one thread per call from D-110): a bit is K threads, each written by its own call; the K calls run in parallel on the
same prompt, which is DPO's pair distribution (independent samples of one prompt). A call returns
one thread (a premise and JOKES lines); the Batch the show reads is joke-major over the threads that came back, and
candidate i of every joke belongs to thread i. The show plays one thread through (app.show._pick_thread).

Failure is the generator's to report and the show's to absorb: a call that fails, times out or returns a bad shape
drops only its thread, and the bit goes on with the rest. `generate` raises, the same exception for every joke of the
bit with no further calls, only when no thread came back, and the show then falls back to the hand-checked bank
(app.show._prepare). The K calls share one wall-time budget (GENERATOR_TIMING[1]), all starting together. Retries inside a call live in app.llm.
"""
import dataclasses
import json
import time
from concurrent.futures import ThreadPoolExecutor, wait

from app import llm, material as material_mod, safety
from app import tonight as tonight_mod
from app.bank import BankGenerator
from app.show import Batch

K = 3  # threads per bit, one call each: enough for the filter to reject some and the selector to still choose
JOKES = 5
TEMPERATURE = 0.9
# The prompt the show runs. gen-v5 (connected bit) won Exp 27 on the 80 fresh prompts (D-111 addendum, 2026-11-26) and went
# live; gen-v4 stays in PROMPTS, frozen byte for byte, as the baseline Exp 26 and Exp 27 measured.
PROMPT_ID = "gen-v5"

# The one reply shape `parse` accepts for a single thread, for providers that can enforce it (Claude structured
# output, Ollama json_schema). gen-v3's three-threads-in-one-reply generator is frozen in app.generate_v3 (D-110).
REPLY_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["premise", "jokes"],
                "properties": {"premise": {"type": "string"},
                               "jokes": {"type": "array", "items": {"type": "string"}}}}
# The same shape with the line count in the grammar. Ollama's json_schema response_format enforces minItems and
# maxItems while it generates, so a small model cannot return 4 or 6 lines (D-111). Claude's structured output takes
# minItems of 0 or 1 only and no maxItems (the SDK moves the rest into the description), so `schema_for` keeps it off
# that path; `parse` enforces the count on both.
COUNTED_REPLY_SCHEMA = {**REPLY_SCHEMA, "properties": {
    **REPLY_SCHEMA["properties"],
    "jokes": {**REPLY_SCHEMA["properties"]["jokes"], "minItems": JOKES, "maxItems": JOKES}}}

# One identity, six attitudes (HumorGen personas as modes, CLAUDE.md invariant 3).
MODES = {
    "neurotic": "anxious and over-analytical: spirals about small things, rehearses worst cases",
    "cynic": "dry and deadpan: unimpressed, resigned, finds the system faintly absurd",
    "observer": "an observational comic: notices the small odd detail of everyday situations and says it plainly",
    "wordsmith": "built on wordplay: puns, double meanings, a phrase taken too literally",
    "optimist": "relentlessly cheerful about things going badly, which is the joke",
    "absurdist": "starts from a normal situation and commits to an escalating, illogical idea",
}
# The structure paragraph is the only part the two prompts differ in (D-111); the mode note, rules, voice and
# reply shape are shared, so a safety or identity edit lands in both.
_HEAD = """You write material for a live stand-up show that one character performs out loud: a self-aware, slightly anxious robot. It is always the same robot; the mode is only the attitude it takes in this bit.

Mode: {mode}: {mode_note}.

"""
_STRUCTURE = """Write one thread of a bit: a premise angle and {jokes} jokes played in order (opener, tag, tag, tag, closer). The opener sets up the situation, each tag builds on it, and the closer calls back to a word or image from this thread's opener. Give the premise as one short sentence. The user message lists premises the room has already heard (premises_already_told): take a clearly different premise from every one of them."""
_STRUCTURE_CONNECTED = """Write one bit of stand-up, not a list of jokes: one premise, and {jokes} lines played in order, each spoken right after the one before it. Line 1 sets up the premise and gets the first laugh. Each later line builds on the line right before it, so the order matters: a tag that adds a twist to the last punchline, then an escalation that pushes it further, then another twist. A line must not start a new topic or work as a standalone one-liner. The last line pays the bit off, with a callback to line 1 or to the premise. Give the premise as one short sentence. The user message lists premises the room has already heard (premises_already_told): take a clearly different premise from every one of them."""
_TAIL = """

Rules:
- Each line is one or two short sentences made for speaking aloud, under 30 words, with no stage directions or emoji.
- Aim every joke at the robot itself, technology, the situation, or things people say. Never at who anyone is: no jokes about bodies or appearance, no protected groups, no real or named people, nobody in the room singled out.
{teens}
Reply with JSON only, exactly this shape: {{"premise": "the premise", "jokes": [opener, tag, tag, tag, closer]}} with {jokes} lines."""
SYSTEM = _HEAD + _STRUCTURE + _TAIL
SYSTEM_CONNECTED = _HEAD + _STRUCTURE_CONNECTED + _TAIL
PROMPTS = {"gen-v4": SYSTEM, "gen-v5": SYSTEM_CONNECTED}
# Appended to the system prompt when material is on (D-117); PROMPTS stay frozen, so this works with any prompt id.
MATERIAL_PARAGRAPH = ("The user message gives this bit's topic and two facts about the robot's life. Build the bit on the "
                      "topic, seen the way this robot sees it; use a fact only where it makes a line funnier. Do not make "
                      "the bit about the robot's own hardware (battery, processor, memory, shutting down).")
MATERIAL_SUFFIX = "+mat-v1"
# The topic-only variant (D-118): its own fixed paragraph; MATERIAL_PARAGRAPH above stays byte-identical (Exp 33's record).
TOPIC_PARAGRAPH = ("The user message gives this bit's topic. Build the bit on the topic, seen the way this robot sees it. "
                   "Do not make the bit about the robot's own hardware (battery, processor, memory, shutting down).")
TOPIC_SUFFIX = "+mat-topic-v1"
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

# Seconds: (per-attempt timeout, deadline for a whole bit over all its calls and attempts), and the retries after the
# first attempt of each call. The bridge lines are sized for that 12 s, so the K calls of a bit share it (D-110).
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
# A local writer's first request loads it into VRAM (Exp 24: 6.1 s for bonsai-2:4b with nothing else loaded, and
# Ollama drops an idle model after 5 minutes by default). Inside a bit that would eat the 12 s deadline, so the show
# pays it once before `show_start`, with the Exp 24 harness's warm-up budget.
WARMUP_TIMEOUT_S = 300.0
WRITER_CONTEXT = 8192  # the context Exp 24 measured the local writers at (D-97's gate); a test holds it to the harness's


# Seconds a call may outlive the bit's budget before the bit stops waiting for it (it still runs to its own timeout).
HANG_GRACE_S = 2.0


class GenerationError(Exception):
    pass


def schema_for(client) -> dict:
    """The counted schema where the client sends it as a grammar (the OpenAI-compatible `json_schema` path), else the
    plain one."""
    return COUNTED_REPLY_SCHEMA if getattr(client, "json_schema", False) else REPLY_SCHEMA


def parse(reply: str) -> tuple[str, list[str]]:
    """Accepts one shape only: {"premise": str, "jokes": [JOKES lines]}. Returns the premise and the lines in playing
    order. Any other shape or count raises, and so does a line with no letters (emoji or symbols only, which the
    voice cannot say): a thread cannot be repaired by guessing."""
    try:
        data = json.loads(llm.strip_fence(reply))
        premise, jokes = data["premise"], data["jokes"]
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise GenerationError(f"malformed reply: {type(e).__name__}") from e
    if (not isinstance(premise, str) or not isinstance(jokes, list) or len(jokes) != JOKES
            or not all(isinstance(line, str) and any(c.isalpha() for c in line) for line in jokes)):
        shape = len(jokes) if isinstance(jokes, list) else type(jokes).__name__
        raise GenerationError(f"expected a premise and {JOKES} lines with letters in them, got {shape}")
    return premise.strip(), [line.strip() for line in jokes]


def parse_reply(reply: str) -> tuple[str, list[str]]:
    """`parse` for a reply that came back from `client.chat(..., validate=parse)`. When the re-ask itself failed (a
    timeout, a 503), the client hands back the first bad reply; the reason says so, so a lost call is not counted as a
    shape error alone (D-111)."""
    try:
        return parse(reply)
    except GenerationError as e:
        err = getattr(reply, "reask_error", None)
        if err is not None:
            raise GenerationError(f"re-ask failed: {type(err).__name__}: {err}") from e
        raise


class ThreadGenerator:
    """One instance per show. A bit is K calls with one identical prompt, run in parallel, one thread each. The
    prompt tells the model which premises the room has heard (only the thread the show actually played counts,
    `played_thread`, not the alternatives that were generated). Two threads of a bit may share a premise: the calls do
    not see each other, as independent samples of one prompt must not (D-110)."""

    material = None  # a subclass that skips __init__ (tests) runs without it
    tonight = None  # a TonightContext (D-119), set by the server for a show with approved context; a subclass that skips __init__ runs without

    def __init__(self, client: llm.Client, *, budget_s: float = GENERATOR_TIMING[1], prompt_id: str | None = None,
                 told: list[str] | None = None, material: "material_mod.ShowMaterial | None" = None):
        self.client = client
        self.material = material
        self.prompt_id = prompt_id or PROMPT_ID
        if self.prompt_id not in PROMPTS:
            raise ValueError(f"unknown prompt {self.prompt_id!r}: expected one of {tuple(PROMPTS)}")
        suffix = "" if not material else TOPIC_SUFFIX if self._topic_only() else MATERIAL_SUFFIX
        self.generator_id = f"{self.prompt_id}{suffix}:{client.model}"
        self.budget_s = budget_s
        self._bits: dict[int, tuple[str, list[list[str]], list[str], list[str]] | Exception] = {}
        self._told: list[str] = list(told or [])  # premises already told: a later bit of a show starts with some
        self._spec: dict[tuple[int, str], tuple | Exception] = {}  # bits written on a guess, not yet adopted
        self._dead: set[tuple[int, str]] = set()  # guesses thrown away: a fetch that ends late writes nothing
        # The material each bit was written with, kept beside the bit (or the guess) so an adopted bit keeps its own.
        self._drawn: dict[int, dict] = {}
        self._spec_drawn: dict[tuple[int, str], dict] = {}
        self._spent: dict[int, dict] = {}  # bit -> material of the thread that was told (played)

    def generate(self, mode: str, bit: int, joke: int, context: dict) -> Batch:
        if bit not in self._bits:
            draw = self._draw(bit, context)
            self._bits[bit] = self._fetch(mode, context, draw, bit) if (draw or self.tonight) else self._fetch(mode, context)
            if draw:
                self._drawn[bit] = draw
        return self._batch(self._bits[bit], mode, bit, joke)

    def _batch(self, got, mode: str, bit: int, joke: int) -> Batch:
        if isinstance(got, Exception):
            raise got
        prompt, lines, _, lost = got
        return Batch(f"{self.prompt_id}:{mode}:{bit}:{joke}", prompt, lines[joke], tuple(lost))

    # A bit prepared on a guess of its mode (D-115) is kept apart, keyed by the mode it was written for, so a wrong guess
    # can neither answer the real request for that bit nor add its premise to the told list. Only `adopt` moves it over.
    def generate_speculative(self, mode: str, bit: int, joke: int, context: dict) -> Batch:
        key = (bit, mode)
        if key not in self._spec:
            draw = self._draw(bit, context)
            got = self._fetch(mode, context, draw, bit) if (draw or self.tonight) else self._fetch(mode, context)  # a worker thread cannot be cancelled: it may end after a discard
            if key in self._dead:
                return self._batch(got, mode, bit, joke)  # nobody will ask again; keep nothing
            self._spec[key] = got
            if draw:
                self._spec_drawn[key] = draw
        return self._batch(self._spec[key], mode, bit, joke)

    def adopt(self, bit: int, mode: str) -> None:
        if (bit, mode) in self._spec:
            self._bits[bit] = self._spec.pop((bit, mode))
            if self.material and (drawn := self._spec_drawn.pop((bit, mode), None)):
                self._drawn[bit] = drawn

    def discard(self, bit: int, mode: str) -> None:
        self._dead.add((bit, mode))
        self._spec.pop((bit, mode), None)
        if self.material:
            self._spec_drawn.pop((bit, mode), None)

    def played_thread(self, bit: int, thread: int) -> None:
        got = self._bits.get(bit)
        if not isinstance(got, Exception) and got and thread < len(got[2]):
            self._told.append(got[2][thread])
            if self.material and (drawn := self._drawn.get(bit)):
                self.material.played(bit, drawn)
                self._spent[bit] = drawn
            if self.tonight:
                self.tonight.played(bit)  # D-119: only a played bit spends its headline

    def material_row(self, bit: int) -> dict | None:
        """The `material` log row's fields for a bit whose thread was played, else None (no material, or a bank bit)."""
        drawn = self._spent.get(bit) if self.material else None
        return None if drawn is None else {**drawn, "seed": self.material.seed}

    def _topic_only(self) -> bool:
        return getattr(self.material, "variant", material_mod.FULL) == material_mod.TOPIC

    def _draw(self, bit: int, context: dict) -> dict | None:
        # A bit that takes an approved headline (D-119) draws no material: its topic would be overwritten, yet it would be
        # spent and logged as if the writer had used it. Judgment call, not measured: such a bit also gets no robot facts.
        if self.tonight and self.tonight.draw(bit):
            return None
        return self.material.draw(bit, context["age"]) if self.material else None

    def set_tonight(self, tonight) -> None:
        self.tonight = tonight
        # Inserted before the first ":" so a material suffix (+mat-v1) set in __init__ survives: prompt+mat+tonight:model
        if tonight_mod.SUFFIX not in self.generator_id:
            self.generator_id = self.generator_id.replace(":", f"{tonight_mod.SUFFIX}:", 1)
        if tonight.allow_people and tonight_mod.PEOPLE_SUFFIX not in self.generator_id:
            self.generator_id = self.generator_id.replace(":", f"{tonight_mod.PEOPLE_SUFFIX}:", 1)

    def tonight_summary(self) -> dict | None:
        return self.tonight.summary() if self.tonight else None

    def tonight_use(self, bit: int) -> dict | None:
        return self.tonight.use_row(bit) if self.tonight else None

    def _fetch(self, mode: str, context: dict, draw: dict | None = None, bit: int | None = None) -> tuple[str, list[list[str]], list[str], list[str]] | Exception:
        """(prompt, joke-major lines, premises, reasons for the threads lost), or the error when no thread came back.
        The error carries the same reasons as `threads_lost`, so the log shows why a bit went to the bank."""
        teens = TEENS_RULE if context["age"] not in safety.STANDARD_AGES else ""
        system = PROMPTS[self.prompt_id].format(mode=mode, mode_note=MODES[mode], jokes=JOKES, teens=teens)
        fields = {"venue": context["venue"], "energy": context["energy"], "premises_already_told": self._told}
        if draw:
            fields["topic"] = draw["topic"]
            if self._topic_only():
                system += "\n\n" + TOPIC_PARAGRAPH
            else:
                system += "\n\n" + MATERIAL_PARAGRAPH
                fields["robot_facts"] = draw["facts"]
        if self.tonight and bit is not None:  # D-119: background for every bit, a headline topic for bits 2 and 6
            system = f"{system}{tonight_mod.SEP}{tonight_mod.PARAGRAPH}"
            if people := self.tonight.people_paragraph():
                system = f"{system}{tonight_mod.SEP}{people}"
            fields |= {k: v for k, v in (("tonight", self.tonight.background()), ("topic", self.tonight.draw(bit))) if v}
        user = json.dumps(fields)
        left = self.budget_s  # the calls start together, so each has the whole bit's budget
        if self.client.deadline_s is not None:
            left = min(left, self.client.deadline_s)
        if left < llm.MIN_ATTEMPT_S:
            return _no_thread(["GenerationError: no time left for a call"])

        def one(_: int):
            try:
                # A bad shape is asked for once more inside this call's deadline (llm.run_with_retries), so one
                # malformed reply costs a second call, not the thread; a second bad reply drops only this thread.
                return parse_reply(self.client.chat(system, user, json_reply=schema_for(self.client), temperature=TEMPERATURE,
                                              validate=parse, deadline_s=left))
            except (llm.LLMError, GenerationError) as e:
                return e  # only this thread is lost

        pool = ThreadPoolExecutor(K)
        futures = [pool.submit(one, i) for i in range(K)]  # read back by call index, not completion order
        done, _ = wait(futures, timeout=left + HANG_GRACE_S)
        pool.shutdown(wait=False, cancel_futures=True)  # a client that ignores its timeout cannot hold the bit
        got: list = []
        for f in futures:
            if f not in done:
                got.append(GenerationError("a call outlived the bit's budget"))
                continue
            try:
                got.append(f.result())
            except Exception as e:  # noqa: BLE001 - not an LLM or shape error, but still only this thread
                got.append(e)
        ok = [g for g in got if not isinstance(g, Exception)]
        lost = [_reason(g) for g in got if isinstance(g, Exception)]
        if not ok:
            return _no_thread(lost)
        return f"{system}\n\n{user}", [[t[1][j] for t in ok] for j in range(JOKES)], [t[0] for t in ok], lost


def _reason(e: Exception) -> str:
    """Why a call produced no thread: the error class and a short message, never joke text (messages hold counts)."""
    return f"{type(e).__name__}: {e}"[:100]


def _no_thread(lost: list[str]) -> GenerationError:
    err = GenerationError(f"no thread written: {lost[-1]}")
    err.threads_lost = lost
    return err


def _ollama_json(url: str, timeout: float = 5.0) -> dict:
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - the operator's own Ollama server
        return json.loads(r.read().decode("utf-8"))


def _tagged(name: str) -> str:
    """Ollama lists a model pulled without a tag as `<name>:latest`; the chat call accepts either."""
    return name if ":" in name else f"{name}:latest"


def placement(client: llm.Client, http=_ollama_json) -> tuple[float | None, int | None, list[str]]:
    """Where Ollama put the writer, from its own `/api/ps`: the share in VRAM, the context it was loaded at, and the
    other models loaded beside it. The share is None when the writer is not listed or the report is implausible or
    cannot be read: unknown, which the show treats as not on the GPU."""
    root = client.base_url.rstrip("/").removesuffix("/v1")
    try:
        loaded = http(f"{root}/api/ps").get("models") or []
    except Exception:
        return None, None, []
    want = _tagged(client.model)
    mine = next((m for m in loaded if want in (_tagged(m.get("name", "")), _tagged(m.get("model", "")))), None)
    others = [m.get("name", "?") for m in loaded if m is not mine]
    size, vram = (mine or {}).get("size"), (mine or {}).get("size_vram")
    if not isinstance(size, (int, float)) or not isinstance(vram, (int, float)) or size <= 0 or not 0 <= vram <= size:
        return None, None, others
    return vram / size, mine.get("context_length"), others


def warm_up(generator, http=_ollama_json) -> float | None:
    """Loads a local writer before the show and checks it is all on the GPU. One untimed request on a copy of its
    client, so its counters and the show's deadlines are untouched. Returns the seconds it took, or None when the
    writer is not local. Raises `llm.LLMError` when the model cannot answer, is not 100% in VRAM or runs at another context (D-109): a writer
    partly on the CPU answers too slowly for the 12 s deadline, and the show would run on the bank alone."""
    client = getattr(generator, "client", None)
    if not getattr(client, "local", False):
        return None
    probe = dataclasses.replace(client, timeout=WARMUP_TIMEOUT_S, deadline_s=WARMUP_TIMEOUT_S, retries=0)
    start = time.monotonic()
    try:
        probe.chat("Reply with JSON only.", '{"ready": true}', json_reply=True)
    except llm.LLMError as e:
        raise llm.LLMError(f"the writer {client.model} did not load ({e}): is `ollama serve` running at "
                           f"{client.base_url}, and is the model pulled (`ollama pull {client.model}`)?", e.status) from e
    took = time.monotonic() - start
    share, context, others = placement(client, http)
    if share is None or share < 1.0:
        where = "Ollama does not list it" if share is None else f"only {share:.0%} of it is on the GPU"
        beside = f"; also loaded: {', '.join(others)} (`ollama stop <name>`)" if others else ""
        raise llm.LLMError(f"the writer {client.model} is not all on the GPU: {where}{beside}. "
                           "Start the show again once it fits (D-97, D-109)")
    if context is not None and context != WRITER_CONTEXT:
        raise llm.LLMError(f"the writer {client.model} is loaded at context {context}, not {WRITER_CONTEXT} as Exp 24 "
                           "measured it: quit the Ollama tray app and restart `ollama serve` with "
                           f"OLLAMA_CONTEXT_LENGTH={WRITER_CONTEXT} (D-97, D-109)")
    return took


class LiveSafety:
    """The show's filter: rules, then one LLM verdict call per bit (`prefetch`), then cached `check`s."""

    def __init__(self, client: llm.Client):
        self.client = client
        self.allow_people: frozenset = frozenset()  # D-120: approved public figures, per show
        self.allow_caps: frozenset = frozenset()  # D-119: the show sets it from tonight's approved gists; empty otherwise

    def _people(self) -> tuple:
        return (self.allow_people,) if self.allow_people else ()

    def prefetch(self, texts: list[str], age: str) -> None:
        safety.prefetch(texts, age, self.client, self.allow_caps, *self._people())

    def check(self, text: str, age: str) -> safety.Verdict:
        return safety.check(text, age, self.allow_caps, *self._people())

    def check_rules(self, text: str, age: str) -> safety.Verdict:
        """The rules alone, for the bank fallback: hand-checked lines need no API call."""
        return safety.check_rules(text, age, self.allow_caps, *self._people())


def writer_client(model: str | None = None) -> llm.Client:
    """The show's joke-writer client, exactly as a show builds it (Exp 26 times this one). `model` names another model on
    the same provider (D-123: the pool tools write with the training base, the eval with each trained writer); the
    request it sends is the live one, only the model field differs."""
    return llm.from_env(GENERATOR_TIMING[0], role="joke_writer", retries=RETRIES, deadline_s=GENERATOR_TIMING[1], model=model)


def _show_material() -> "material_mod.ShowMaterial | None":
    return (material_mod.ShowMaterial(variant=v) if (v := material_mod.variant()) else None)


def llm_stack(room: str):
    """(generator, safety, fallback, bridge lines) for a show on the real model. Raises SafetyNotReady for a live
    room without the slur list. The deadlines keep a failing provider inside the bridge lines' cover."""
    safety.require_ready(room)
    return (ThreadGenerator(writer_client(), material=_show_material()),
            LiveSafety(llm.from_env(FILTER_TIMING[0], role="safety_check", retries=FILTER_RETRIES, deadline_s=FILTER_TIMING[1])),
            BankGenerator(), list(BRIDGES))
