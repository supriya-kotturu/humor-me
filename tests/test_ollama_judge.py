"""The Ollama option for the Exp 24 judge: the guards against Ollama's silent prompt truncation (on the cluster script and on
the laptop's /v1 client), the thinking-model parser, and the judge running through the guard or from a bundle. No server."""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from app import llm  # noqa: E402
from explorer import judge_generators as ej  # noqa: E402
from sim import compare_generators as cg  # noqa: E402
from sim import judge_generators as jg  # noqa: E402
from tests.test_compare_generators import MODELS, NoWait, data, fresh_safety  # noqa: E402,F401
from tests.test_judge_generators import PREF, prep, rank_by  # noqa: E402

LETTERS = ["A", "B", "C"]


def ollama_call():
    return {"id": "c0", "system": "s" * 1000, "user": "u" * 40000, "letters": LETTERS}


class FakeOllama:
    """The native /api/chat: reports `read_share` of the sent prompt as evaluated (below 1 simulates truncation)."""

    def __init__(self, content='{"ranking": "BAC"}', read_share=1.0, no_count=False):
        self.content, self.read_share, self.no_count, self.sent = content, read_share, no_count, []

    def __call__(self, payload):
        self.sent.append(payload)
        out = {"message": {"role": "assistant", "content": self.content}}
        if not self.no_count:
            out["prompt_eval_count"] = int(ej.estimate_tokens(payload["messages"]) * self.read_share)
        return out


# --- the cluster script ---

def test_ollama_is_asked_for_the_full_context_greedy_without_thinking_and_with_a_unique_start():
    post = FakeOllama()
    conv = ej.conv(ollama_call())
    assert ej.ollama_reply(post, "qwen3.8:27b", 32768, conv) == '{"ranking": "BAC"}'
    ej.ollama_reply(post, "qwen3.8:27b", 32768, conv)
    p = post.sent[0]
    assert p["model"] == "qwen3.8:27b" and p["stream"] is False and p["think"] is False
    assert p["options"]["num_ctx"] == 32768 and p["options"]["temperature"] == 0
    s0, s1 = post.sent[0]["messages"][0]["content"], post.sent[1]["messages"][0]["content"]
    assert s0 != s1 and s0.startswith("s" * 1000) and "(request " in s0  # no cached prefix: prompt_eval_count is the whole prompt
    assert conv[0]["content"] == "s" * 1000  # the caller's messages are untouched


def test_ollama_refuses_a_prompt_over_the_context_without_calling_the_server():
    post = FakeOllama()
    out = ej.ollama_reply(post, "m", 8000, ej.conv(ollama_call()))  # about 11.7k estimated tokens
    assert out.startswith("[refused") and post.sent == [] and ej.parse_ranking(out, LETTERS) is None


@pytest.mark.parametrize("post", [FakeOllama(read_share=0.3), FakeOllama(no_count=True)])
def test_ollama_reading_far_fewer_tokens_than_sent_is_truncation_not_an_answer(post, tmp_path):
    out = ej.ollama_reply(post, "m", 32768, ej.conv(ollama_call()))
    assert out.startswith("[truncated") and ej.parse_ranking(out, LETTERS) is None
    gen = lambda convs: [ej.ollama_reply(post, "m", 32768, c) for c in convs]  # noqa: E731
    counts = ej.answer_calls([ollama_call()], gen, tmp_path / "r.jsonl", log=lambda m: None)
    assert counts["answered"] == 0 and counts["failed"] == 1
    failed = json.loads((tmp_path / "r.jsonl.failed.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert failed["raw"].startswith("[truncated")  # the failed file says why, and the call is retried next run
    assert not (tmp_path / "r.jsonl").exists() or (tmp_path / "r.jsonl").read_text(encoding="utf-8") == ""


def test_the_tolerant_parser_drops_a_thinking_models_reasoning():
    assert ej.parse_ranking('<think>A is flat, maybe CAB</think>\n{"ranking": "BAC"}', LETTERS) == "BAC"
    assert ej.parse_ranking("<think>I will rank C first, then A, then B", LETTERS) is None  # cut off mid-thought
    assert ej.parse_ranking("<think>x</think>CBA", LETTERS) == "CBA"


def test_the_ollama_generator_runs_through_the_same_answer_loop(tmp_path):
    class Args:
        model, context, url = "qwen3.8:27b", 32768, "http://x"

    gen = ej.make_ollama_generator(Args, post=FakeOllama())
    counts = ej.answer_calls([ollama_call()], gen, tmp_path / "r.jsonl", log=lambda m: None)
    assert counts["answered"] == 1 and json.loads((tmp_path / "r.jsonl").read_text(encoding="utf-8"))["ranking"] == "BAC"


# --- the laptop side ---

class InnerOllama:
    """The llm.Client talking to Ollama's /v1: reports prompt tokens as `read_share` of what it was sent."""
    model = "qwen3.8:27b"

    def __init__(self, read_share=1.0):
        self.read_share, self.input_tokens, self.output_tokens, self.systems = read_share, 0, 0, []

    def chat(self, system, user, *, json_reply=False, temperature=0.0, validate=None):
        self.systems.append(system)
        self.input_tokens += int(jg.estimate_tokens(system, user) * self.read_share)
        return json.dumps({"ranking": rank_by(user, lambda t: PREF[t])})


def test_the_ollama_judge_has_a_filesystem_safe_key_and_the_models_name():
    spec = jg.resolve_judge(None, "ollama")
    assert (spec.model, spec.provider) == ("qwen3.8:27b", "ollama") and ":" not in spec.key and spec.key == "ollama-qwen3.8-27b"
    assert jg.resolve_judge("llama3.1:70b", "ollama").key == "ollama-llama3.1-70b"
    assert jg.resolve_judge(None, None).key == jg.JUDGE_MODEL  # Qwen2.5-7B on Explorer stays the default
    assert jg.resolve_judge("claude-haiku-5-5", "anthropic").provider == "anthropic"
    with pytest.raises(SystemExit):
        jg.resolve_judge("nope", None)


def test_the_guard_turns_a_truncated_reply_into_an_infrastructure_failure():
    guard = jg.OllamaGuard(InnerOllama(read_share=0.25), context=32768)
    with pytest.raises(llm.LLMError) as e:
        guard.chat("s" * 1000, "u" * 40000 + "\n## New setting to rank\nOption A:\n1. alphav1 thread 0")
    assert e.value.retryable and cg.failure_kind(e.value) == "infra"
    a = InnerOllama()
    g = jg.OllamaGuard(a, context=32768)
    user = "## New setting to rank\nOption A:\n1. alphav1 thread 0\nOption B:\n1. bravov1 thread 0"
    assert json.loads(g.chat("sys", user))["ranking"] == "BA"
    g.chat("sys", user)
    assert a.systems[0] != a.systems[1] and all(s.startswith("sys") for s in a.systems)  # a unique start defeats prefix caching
    silent = type("C", (), {"model": "m", "input_tokens": 0, "output_tokens": 0, "chat": lambda self, *a, **k: "{}"})()
    with pytest.raises(llm.LLMError, match="truncated, or no usage"):
        jg.OllamaGuard(silent, context=32768).chat("s", "u" * 4000)


def test_the_guard_and_the_run_refuse_a_prompt_over_the_context(data, monkeypatch):
    with pytest.raises(llm.LLMError, match="over the 1000-token context"):
        jg.OllamaGuard(InnerOllama(), context=1000).chat("s", "u" * 40000)
    d, key, votes = prep()
    spec = jg.resolve_judge(None, "ollama")
    monkeypatch.setenv("OLLAMA_CONTEXT_LENGTH", "2000")
    tiny = InnerOllama()
    with pytest.raises(SystemExit, match="context is 2000"):
        jg.run_judge(MODELS, 60, 30, lambda s: jg.OllamaGuard(tiny), spec.key, directory=d, pacer=NoWait(), log=lambda s: None, spec=spec)
    assert tiny.systems == []  # nothing was sent: the judge never ranks bits it was not shown
    monkeypatch.delenv("OLLAMA_CONTEXT_LENGTH")
    assert jg.context_length() == 32768


def test_an_ollama_judge_through_the_guard_scores_and_a_truncating_server_leaves_the_run_incomplete(data, monkeypatch):
    monkeypatch.setenv("OLLAMA_CONTEXT_LENGTH", "32768")
    d, key, votes = prep()
    spec = jg.resolve_judge(None, "ollama")

    def run_(inner):
        return jg.run_judge(MODELS, 60, 30, lambda s: jg.OllamaGuard(inner), spec.key, directory=d, pacer=NoWait(),
                            log=lambda s: None, spec=spec)

    cut = run_(InnerOllama(read_share=0.2))
    assert cut["incomplete"] and cut["n_unanswered"] == cut["n_calls"] and not cut["joined"]
    assert not (d / "judge_cache" / spec.key).exists() or not list((d / "judge_cache" / spec.key).iterdir())  # never cached
    ok = run_(InnerOllama())
    assert not ok["incomplete"] and ok["joined"] and ok["agreement"]["rate"] == 1.0


def test_the_ollama_judge_runs_from_a_bundle_too(data):
    d, key, votes = prep()
    spec = jg.resolve_judge(None, "ollama")
    out = d.parent / "judge-ollama"
    assert jg.export_bundle(MODELS, 60, 30, spec.key, out, directory=d)["calls"] == 180
    rows = [json.loads(line) for line in (out / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(":" not in r["id"] for r in rows)
    replies = d.parent / "replies.jsonl"
    replies.write_text("".join(json.dumps({"id": r["id"], "ranking": rank_by(r["user"], lambda t: PREF[t])}) + "\n" for r in rows), encoding="utf-8")
    assert jg.import_replies(replies, spec.key, d)["imported"] == 180
    res = jg.run_judge(MODELS, 60, 30, lambda s: jg.ReplayClient(), spec.key, directory=d, pacer=NoWait(), log=lambda s: None, spec=spec)
    assert res["joined"] and res["judge_model"] == "qwen3.8:27b"


def test_a_thinking_models_reply_is_parsed_after_its_reasoning():
    assert jg._parse('<think>hmm, B maybe</think>{"ranking": "BAC"}', {"A": "x", "B": "y", "C": "z"}) == "BAC"
