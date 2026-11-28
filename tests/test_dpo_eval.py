"""Exp 38 (D-123): the vote's bookkeeping, the power table behind the proposed bar, the judges' tallies and the latency summary, with
stored fake bits and a fake judge. No model, no network."""
import json
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app import generate  # noqa: E402
from sim import dpo_eval as de  # noqa: E402
from sim import dpo_pairs as dp  # noqa: E402
from sim import dpo_train as dt  # noqa: E402
from tests.test_dpo_pairs import day, scripted  # noqa: E402
from tests.test_dpo_train import teach_pairs  # noqa: E402


@pytest.fixture(scope="module")
def prompts():
    return dt.build_eval_prompts(dp.build_pool([day()]))


def eval_bit(p, writer, wall=5.0, clean=True, threads=3):
    key = {"A": "a", "B": "b", "C": "c"}[next(k for k, w in de.WRITERS.items() if w == writer)]
    ts = [{"premise": "x", "lines": [f"{p['id']} {key} t{i} line {j}." for j in range(5)], "safe": clean} for i in range(threads)]
    return {**{k: p[k] for k in dp.PROMPT_FIELDS}, "id": p["id"], "writer": writer, "threads": ts, "safety_infra_error": False,
            "digest": "d", "wall_s": wall}


def fill(root, prompts, skip=(), walls=None):
    dt.write_jsonl(root / "train" / "eval_prompts.jsonl", prompts)
    for p in prompts:
        for k, w in de.WRITERS.items():
            if (p["id"], k) not in skip:
                dp._write(dp.bit_path(root, "eval", p["id"], w), eval_bit(p, w, wall=(walls or {}).get(k, 5.0)))


@pytest.fixture
def root(tmp_path, prompts):
    fill(tmp_path, prompts)
    return tmp_path


# --- the power table and the proposed bar --------------------------------------------------------------------------------

def test_the_power_table_is_pinned_and_the_proposed_bar_is_a_coin_flip_at_15_percent():
    t = de.power_table()
    assert t[10][0.5] == pytest.approx(0.1509, abs=1e-4) and t[11][0.5] == pytest.approx(0.0592, abs=1e-4)
    assert t[10][0.65] == pytest.approx(0.5643, abs=1e-4) and t[10][0.7] == pytest.approx(0.7216, abs=1e-4)
    assert t[10][0.75] == pytest.approx(0.8516, abs=1e-4)
    assert de.either_passes(10) == pytest.approx(0.279, abs=1e-3) and de.either_passes(11) == pytest.approx(0.115, abs=1e-3)
    assert de.p_at_least(0, 15, 0.3) == pytest.approx(1.0) and de.p_at_least(15, 15, 1.0) == 1.0
    assert de.power_ties()[12][0.5] == pytest.approx(0.0193, abs=1e-3)  # ties shrink the decided pairs: harder, not softer
    assert "at least 10" in de.power_text() and de.PROPOSED_WINS == 10


# --- the plan ------------------------------------------------------------------------------------------------------------

def test_the_plan_is_15_pairs_per_comparison_on_different_prompts_seeded_and_mixed_by_half(root, prompts):
    items = de.plan(prompts, root)
    assert len(items) == 30 and de.plan(prompts, root) == items  # seeded
    assert [sum(i["comp"] == c for i in items) for c in de.VOTED] == [15, 15]
    assert len({i["pid"] for i in items}) == 30  # a prompt is used once
    half = {p["id"]: p["half"] for p in prompts}
    for c in de.VOTED:
        n = sum(half[i["pid"]] == "tonight" for i in items if i["comp"] == c)
        assert 6 <= n <= 9  # both halves, nearly even
    assert 8 <= sum(i["trained_left"] for i in items) <= 22  # sides are mixed
    for i in items:  # the trained writer's lines are on its side, the base's on the other
        trained = "b" if i["comp"] == "B" else "c"
        left_is_trained = f" {trained} " in i["left"][0]
        assert left_is_trained == i["trained_left"] and (" a " in (i["right"] if left_is_trained else i["left"])[0])


def test_the_plan_skips_a_prompt_where_a_writer_has_no_clean_thread_and_comes_up_short_when_it_must(tmp_path, prompts):
    fill(tmp_path, prompts, skip={(prompts[0]["id"], "B"), (prompts[1]["id"], "A")})
    items = de.plan(prompts, tmp_path)
    used = {(i["pid"], i["comp"]) for i in items}
    assert len(items) == 30 and (prompts[0]["id"], "B") not in used and prompts[1]["id"] not in {pid for pid, _ in used}
    few = tmp_path / "few"
    fill(few, prompts[:20])
    assert len(de.plan(prompts[:20], few)) == 20  # not enough prompts for 30: `vote` refuses until more are written


# --- the vote ------------------------------------------------------------------------------------------------------------

def test_vote_refuses_until_the_bar_is_fixed_and_the_bar_cannot_move_after_a_vote(root, prompts):
    items = de.plan(prompts, root)
    with pytest.raises(SystemExit, match="fix the bar"):
        de.vote(items, root, read=scripted("1"), out=lambda *_: None)
    assert de.set_bar(root, 10)["wins"] == 10 and de.read_bar(root)["of"] == 15 and "at" in de.read_bar(root)
    with pytest.raises(SystemExit):
        de.set_bar(root, 16)
    de.vote(items, root, limit=1, read=scripted("1"), out=lambda *_: None)
    with pytest.raises(SystemExit, match="cannot change"):
        de.set_bar(root, 9)


def test_vote_saves_each_answer_with_a_timestamp_allows_ties_never_a_skip_and_resumes(root, prompts):
    de.set_bar(root, 10)
    items = de.plan(prompts, root)
    out = []
    n = de.vote(items, root, read=scripted("s", "x", "1", "2", "t", "q"), out=out.append)
    assert n == 3  # s and x are asked again, not taken
    state = de.read_votes(root)
    assert [a["choice"] for a in state["answers"].values()] == ["1", "2", "t"]
    assert all(re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", a["at"]) for a in state["answers"].values())
    assert list(state["answers"]) == [de.vote_key(i) for i in items[:3]]  # in presentation order
    shown = "\n".join(out)
    assert "humor-qwen3" not in shown and "qwen3" not in shown and "wins" not in shown and "Pair 1 of 30" in shown
    assert de.vote(items, root, read=scripted(*["1"] * 40), out=lambda *_: None) == 30  # resumed: only the 27 left were asked
    assert len(de.read_votes(root)["answers"]) == 30


def test_vote_refuses_when_fewer_than_30_pairs_are_ready(tmp_path, prompts):
    fill(tmp_path, prompts[:20])
    de.set_bar(tmp_path, 10)
    with pytest.raises(SystemExit, match="of 30 pairs are ready"):
        de.vote(de.plan(prompts[:20], tmp_path), tmp_path, read=scripted("1"), out=lambda *_: None)


def test_the_tally_counts_a_win_for_the_trained_writer_on_either_side_and_applies_the_bar(root, prompts):
    de.set_bar(root, 10)
    items = de.plan(prompts, root)
    answers = {}
    for k, i in enumerate(items):
        trained_side = "1" if i["trained_left"] else "2"
        base_side = "2" if i["trained_left"] else "1"
        choice = "t" if k == 0 else trained_side if i["comp"] == "B" or k % 2 else base_side  # B: always trained; C: half
        answers[de.vote_key(i)] = {"choice": choice, "trained_left": i["trained_left"], "at": "x"}
    res = de.tally(items, {"answers": answers}, de.read_bar(root))
    b, c = res["B"], res["C"]
    assert b["answered"] == c["answered"] == 15 and b["wins"] + b["losses"] + b["ties"] == 15
    assert b["wins"] == 15 - b["ties"] and c["wins"] + c["losses"] + c["ties"] == 15
    assert b["verdict"] == ("met" if b["wins"] >= 10 else "not met") and c["verdict"] == ("met" if c["wins"] >= 10 else "not met")
    half = dict(list(answers.items())[:10])
    assert de.tally(items, {"answers": half}, de.read_bar(root))["B"]["verdict"] == "pending"
    assert de.tally(items, {"answers": answers}, None)["B"]["verdict"] == "pending"  # no bar fixed: nothing is met


# --- the judges (reported only) ------------------------------------------------------------------------------------------

class FakeJudge:
    """Prefers writer B over C over A (by the letter in the line) in either order; `bias` answers A every time."""
    rank = {"b": 0, "c": 1, "a": 2}

    def __init__(self, bias=False):
        self.calls, self.bias = [], bias
        self.input_tokens = self.output_tokens = 0

    def chat(self, system, user, *, json_reply=False, validate=None, **kw):
        self.calls.append(user)
        a, b = re.findall(r"e\d{4} ([abc]) t", user)[::5][:2]
        return json.dumps({"funnier": "A" if self.bias or self.rank[a] < self.rank[b] else "B"})


def test_the_judge_asks_all_three_comparisons_in_both_orders_resumes_and_tallies(root, prompts):
    pairs = teach_pairs()
    judge = FakeJudge()
    n = de.judge_all(judge, pairs, prompts, root, "opus", out=lambda *_: None)
    assert n == len(prompts) * 3 * 2 == len(judge.calls)
    assert de.judge_all(FakeJudge(), pairs, prompts, root, "opus", out=lambda *_: None) == 0
    t = de.judge_tally(pairs, prompts, root, "opus")
    n_p = len(prompts)
    assert t["BA"] == {"consistent": n_p, "first_wins": n_p, "flips": 0, "missing": 0}
    assert t["CA"]["first_wins"] == n_p and t["BC"]["first_wins"] == n_p
    biased = FakeJudge(bias=True)
    de.judge_all(biased, pairs, prompts, root, "haiku", out=lambda *_: None)
    h = de.judge_tally(pairs, prompts, root, "haiku")
    assert all(v["consistent"] == 0 and v["flips"] == n_p for v in h.values())  # position bias: the orders disagree
    assert de.judge_dir(root, dp.teach_prompt(pairs), "opus") != de.judge_dir(root, dp.teach_prompt(pairs), "haiku")


def test_a_comparison_is_judged_only_where_both_writers_have_a_clean_thread(tmp_path, prompts):
    fill(tmp_path, prompts, skip={(prompts[0]["id"], "B")})
    judge = FakeJudge()
    de.judge_all(judge, teach_pairs(), prompts, tmp_path, "opus", out=lambda *_: None)
    assert len(judge.calls) == (len(prompts) * 3 - 2) * 2  # B is missing for one prompt: B-A and B-C skip it
    t = de.judge_tally(teach_pairs(), prompts, tmp_path, "opus")
    assert t["BA"]["consistent"] == len(prompts) - 1 and t["CA"]["consistent"] == len(prompts)


# --- latency -------------------------------------------------------------------------------------------------------------

def test_latency_reports_median_and_slowest_5_percent_against_the_budget(tmp_path, prompts):
    fill(tmp_path, prompts)
    for i, p in enumerate(prompts):  # B: the slowest 5% (3 of 60 bits) run over the budget; C: all within
        dp._write(dp.bit_path(tmp_path, "eval", p["id"], de.WRITERS["B"]), eval_bit(p, de.WRITERS["B"], wall=13.0 if i < 4 else 6.0 + i / 100))
        dp._write(dp.bit_path(tmp_path, "eval", p["id"], de.WRITERS["C"]), eval_bit(p, de.WRITERS["C"], wall=7.0 + i / 100))
    res = de.latency(prompts, tmp_path)
    assert res["budget_s"] == generate.GENERATOR_TIMING[1] == 12.0
    a, b, c = (res["writers"][k] for k in "ABC")
    assert a["p50_s"] == a["p95_s"] == 5.0 and a["within_budget"] and a["bits"] == 60
    assert b["p95_s"] == 13.0 and not b["within_budget"] and b["max_s"] == 13.0
    assert c["within_budget"] and c["p50_s"] == pytest.approx(7.3, abs=0.05) and c["p95_s"] == pytest.approx(7.57, abs=0.05)
    assert "bonsai_bit_p50_s" in res and "bonsai_bit_p95_s" in res  # Exp 26's numbers, from the committed record
    assert res["bar"] == de.LATENCY_BAR


def test_a_bit_that_lost_a_thread_is_counted(tmp_path, prompts):
    fill(tmp_path, prompts)
    dp._write(dp.bit_path(tmp_path, "eval", prompts[0]["id"], de.WRITERS["C"]), eval_bit(prompts[0], de.WRITERS["C"], threads=2))
    assert de.latency(prompts, tmp_path)["writers"]["C"]["bits_losing_a_thread"] == 1


# --- score and the record ------------------------------------------------------------------------------------------------

def test_score_is_pending_before_the_votes_and_done_after(root, prompts, monkeypatch):
    monkeypatch.setattr(dp, "need_pairs", lambda r=None: ({}, {}, teach_pairs()))
    res = de.score(root)
    assert res["status"] == "pending" and res["votes"]["B"]["verdict"] == "pending" and res["proposed_wins"] == 10
    assert "pending" in de.table(res)
    de.set_bar(root, 10)
    de.vote(de.plan(prompts, root), root, read=scripted(*["1"] * 30), out=lambda *_: None)
    done = de.score(root)
    assert done["status"] == "done" and done["bar"]["wins"] == 10
    text = de.table(done)
    assert "B against A, your blind votes" in text and "within the 12 s budget" in text
    p = de.params(root)
    assert p["bar"] == 10 and p["prompts_digest"] and set(p["bits"]) == {"A", "B", "C"}


def test_the_eval_bits_are_written_through_the_pool_writer_into_the_writers_own_folder(tmp_path, prompts):
    from tests.test_dpo_pairs import FakeOllama, FakeSafety
    from tests.test_dpo_train import client_for
    pool = {"days": [day()], "prompts": prompts}
    assert dp.write("eval", client_for("humor-qwen3-B", FakeOllama()), FakeSafety(), pool, tmp_path, n=2, out=lambda *_: None) == 2
    rec = dp._read(dp.bit_path(tmp_path, "eval", prompts[0]["id"], "humor-qwen3-B"))
    assert (tmp_path / "bits" / "eval-humor-qwen3-B").is_dir() and rec["writer"] == "humor-qwen3-B" and rec["split"] == "eval"
    assert de.bit_of(tmp_path, "humor-qwen3-B", prompts[0]["id"]) is not None and de.bit_of(tmp_path, de.WRITERS["A"], prompts[0]["id"]) is None
