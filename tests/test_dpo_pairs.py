"""Exp 36 (D-121): the pair pool, the background writer, the ranking tool and the judge gate's bookkeeping, with a fake
Ollama, fake safety, scripted answers and a fake judge. No model, no network."""
import collections
import itertools
import json
import pathlib
import sys
import threading

import httpx
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app import generate, llm, safety  # noqa: E402
from app import tonight as app_tonight  # noqa: E402
from sim import dpo_pairs as dp  # noqa: E402
from sim import judge_calibration as jc  # noqa: E402


# --- fakes ---------------------------------------------------------------------------------------------------------------

def day(date="2026-11-26", gists=("a chip appears", "a vote happens", "a parade moves"), people=None):
    return {"date": date, "approved": list(gists), "facts": [], "market": "The S&P 500 fell 0.5% on Nov 24.",
            "names": [], "people": list(people or [])}


class FakeOllama:
    """Answers a writer request with a thread of unique lines; a thread whose number is in `bad` carries the word BAD in
    one line, which FakeSafety rejects. Records every request body."""

    def __init__(self, bad=()):
        self.seen, self.lock, self.n, self.bad = [], threading.Lock(), 0, set(bad)

    def __call__(self, request):
        body = json.loads(request.content)
        with self.lock:
            self.seen.append(body)
            self.n += 1
            n = self.n
        lines = [f"thread {n} line {j}." for j in range(5)]
        if n in self.bad:
            lines[2] = f"thread {n} BAD line."
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(
            {"premise": f"premise {n}", "jokes": lines})}, "finish_reason": "stop"}]})

    def client(self):
        return llm.Client("http://local/v1", "", "bonsai-2:4b", 10.0, retries=2, deadline_s=12.0, local=True,
                          extra_body=llm.OLLAMA_NO_THINK, json_schema=True, transport=httpx.MockTransport(self))


class FakeSafety:
    """Rejects a line with BAD; `fail` makes its own layer report an outage; remembers the allow lists the writer set."""

    def __init__(self, fail=False):
        self.fail, self.allow_caps, self.allow_people, self.checked = fail, frozenset(), frozenset(), 0

    def prefetch(self, texts, age):
        pass

    def check(self, text, age):
        self.checked += 1
        if self.fail:
            return safety.Verdict(False, "llm:error TimeoutError")
        return safety.Verdict(False, "rule:test") if "BAD" in text else safety.Verdict(True, "ok")


@pytest.fixture(scope="module")
def pool():
    return dp.build_pool([day()])


def fake_bits(pool, root, n_threads=3):
    """A written, clean bit for every ranking prompt, without a model."""
    for p in pool["prompts"]:
        if p["split"] == "ranking":
            threads = [{"premise": "p", "lines": [f"{p['id']} t{i} line {j}." for j in range(5)], "safe": True, "rejected": []}
                       for i in range(n_threads)]
            dp._write(dp.bit_path(root, "ranking", p["id"]), {"id": p["id"], "threads": threads, "safety_infra_error": False})


def scripted(*answers):
    it = iter(answers)

    def read(prompt):
        return next(it, "q")
    return read


# --- the pool ------------------------------------------------------------------------------------------------------------

def test_the_pool_is_deterministic_and_the_splits_are_disjoint(pool):
    again = dp.build_pool([day()])
    assert again == pool  # same seed, same data, same pool
    keys = {s: {dp.prompt_key(p) for p in pool["prompts"] if p["split"] == s} for s in ("ranking", "train")}
    assert not keys["ranking"] & keys["train"]
    assert not (keys["ranking"] | keys["train"]) & dp.earlier_keys()  # Exp 24's ballot, 26, 27, 30
    assert len({p["id"] for p in pool["prompts"]}) == len(pool["prompts"])
    dp.assert_disjoint(pool)


def test_a_prompt_equal_to_an_earlier_experiments_is_caught(pool):
    bad = json.loads(json.dumps(pool))
    from sim import compare_generators as cg
    p24 = cg.prompts(65)[0]
    victim = next(p for p in bad["prompts"] if p["half"] == "plain" and p["split"] == "train")
    victim.update(mode=p24["mode"], context=p24["context"], told=[], bit=0)
    with pytest.raises(AssertionError, match="Exp 24"):
        dp.assert_disjoint(bad)


def test_an_identical_tonight_request_at_another_bit_number_is_caught(pool):
    # Bits 4 to 7 all have 4 premises told and, off the headline bits, the same request: the bit number is not in it.
    bad = json.loads(json.dumps(pool))
    r = next(p for p in bad["prompts"] if p["split"] == "ranking" and p["half"] == "tonight" and p["bit"] == 4)
    t = next(p for p in bad["prompts"] if p["split"] == "train" and p["half"] == "tonight" and p["bit"] == 5)
    t.update(mode=r["mode"], context=dict(r["context"]), told=list(r["told"]))
    with pytest.raises(AssertionError, match="ranking prompt equals a train prompt"):
        dp.assert_disjoint(bad)


def test_no_headline_gist_is_in_both_a_teach_and_a_test_prompt():
    many = dp.build_pool([day(gists=[f"gist {i}" for i in range(21)])])  # as stored: 21 gists for 30 headline prompts
    roles = collections.defaultdict(set)
    for p in many["prompts"]:
        if p["split"] == "ranking" and p["gist"]:
            roles[p["gist"]].add(p["role"])
    shared = [g for g in roles if sum(q["gist"] == g for q in many["prompts"] if q["split"] == "ranking") > 1]
    assert shared and all(len(r) == 1 for r in roles.values())
    assert all(sum(p["split"] == "ranking" and (p["role"], p["half"]) == c for p in many["prompts"]) == 60 for c in dp.CELLS)


def test_sizes_halves_roles_and_coverage(pool):
    assert dp.sizes(pool) == {"ranking/tonight": 120, "ranking/plain": 120, "train/tonight": 400, "train/plain": 400}
    ranking = [p for p in pool["prompts"] if p["split"] == "ranking"]
    for cell in dp.CELLS:
        mine = [p for p in ranking if (p["role"], p["half"]) == cell]
        assert len(mine) == 60  # 30 decided pairs from 60 prompts leaves room for ties, skips and thin bits
    assert all(p["role"] is None for p in pool["prompts"] if p["split"] == "train")
    for s in ("ranking", "train"):
        for h in dp.HALVES:
            mine = [p for p in pool["prompts"] if p["split"] == s and p["half"] == h]
            assert {p["mode"] for p in mine} == set(generate.MODES)
            assert {p["context"]["venue"] for p in mine} == set(dp.VENUES)
            assert {p["context"]["energy"] for p in mine} == set(dp.ENERGIES)
            assert {p["context"]["age"] for p in mine} == set(dp.AGES)
            assert {p["bit"] for p in mine} == set(range(8))


def test_roles_are_fixed_by_seed_before_any_answer(pool):
    assert [p["role"] for p in dp.build_pool([day()])["prompts"]] == [p["role"] for p in pool["prompts"]]
    assert dp.RANK_SEED in json.dumps(pool["rank_seed"])


def test_gists_go_round_robin_on_headline_bits_only(pool):
    tonight = [p for p in pool["prompts"] if p["half"] == "tonight"]
    with_gist = [p for p in tonight if p["gist"]]
    assert {p["bit"] for p in with_gist} == {2, 6}
    assert all(p["bit"] in (2, 6) for p in with_gist) and all(p["gist"] for p in tonight if p["bit"] in (2, 6))
    counts = collections.Counter(p["gist"] for p in with_gist)
    assert set(counts) == set(day()["approved"]) and max(counts.values()) - min(counts.values()) <= 1
    assert all(p["gist"] is None and p["day"] is None for p in pool["prompts"] if p["half"] == "plain")


def test_a_pool_without_approved_gists_stops():
    with pytest.raises(SystemExit, match="approved gists"):
        dp.build_pool([])


def test_load_days_takes_the_newest_fetch_of_a_date_and_only_approved_gists(tmp_path):
    def put(name, fetched, approved):
        dp._write(tmp_path / name / "fetch.json", {"date": "2026-11-26", "fetched_at": fetched, "approved": approved,
                                                    "answers": [{"gist": g, "approved": g in approved} for g in ("a", "b", "c")],
                                                    "market": {"line": "up"}, "facts": [], "keep": {"b": ["AI"]}, "people": {}})
    put("old", "2026-11-26T10:00", ["a"])
    put("new", "2026-11-27T01:00", ["b", "c"])
    put("none", "2026-11-28T01:00", [])
    got = dp.load_days(tmp_path)
    assert [(d["date"], d["approved"], d["names"]) for d in got] == [("2026-11-26", ["b", "c"], ["AI"])]


# --- the request is the show's -------------------------------------------------------------------------------------------

def show_request(told, bit, d, headline=None):
    """What a show with this day's approved context sends at `bit`, built the way the server builds it."""
    fake = FakeOllama()
    gen = generate.ThreadGenerator(fake.client(), told=told)
    gen.set_tonight(app_tonight.TonightContext(d["facts"], d["market"], d["approved"], allowed_names=d["names"], allowed_people=d["people"]))
    gen.generate("cynic", bit, 0, {"venue": "bar", "energy": "high", "age": "young_adults"})
    return fake.seen[0]["messages"], gen.generator_id


def pool_request(prompt, d):
    fake = FakeOllama()
    gen = dp.new_generator(fake.client(), prompt, {d["date"]: d})
    gen.generate(prompt["mode"], prompt["bit"], 0, prompt["context"])
    return fake.seen[0]["messages"], gen.generator_id


@pytest.mark.parametrize("bit", [2, 3])
@pytest.mark.parametrize("people", [[], ["Jane Roe"]])
def test_the_tonight_half_builds_the_request_the_show_builds(bit, people):
    d = day(people=people)
    told = ["a tax form"]
    prompt = {"id": "x", "split": "ranking", "half": "tonight", "mode": "cynic", "bit": bit, "day": d["date"],
              "gist": d["approved"][0] if bit == 2 else None, "told": told,
              "context": {"venue": "bar", "energy": "high", "age": "young_adults"}}
    mine, mine_id = pool_request(prompt, d)
    theirs, their_id = show_request(told, bit, d)
    assert mine == theirs and mine_id == their_id
    user = json.loads(mine[1]["content"])
    assert user.get("topic") == (d["approved"][0] if bit == 2 else None)
    assert user["tonight"] == {"facts": [], "market": d["market"]} and app_tonight.SUFFIX in their_id


def test_the_plain_half_is_the_live_prompt_with_no_tonight_and_no_material():
    prompt = {"id": "x", "split": "train", "half": "plain", "mode": "cynic", "bit": 3, "day": None, "gist": None, "told": ["a tax form"],
              "context": {"venue": "bar", "energy": "high", "age": "teens"}}
    fake = FakeOllama()
    gen = dp.new_generator(fake.client(), prompt, {})
    gen.generate("cynic", 3, 0, prompt["context"])
    system, user = fake.seen[0]["messages"][0]["content"], json.loads(fake.seen[0]["messages"][1]["content"])
    assert gen.generator_id == f"{generate.PROMPT_ID}:bonsai-2:4b"
    assert set(user) == {"venue", "energy", "premises_already_told"} and app_tonight.PARAGRAPH not in system
    assert user["premises_already_told"] == ["a tax form"]


# --- the writer ----------------------------------------------------------------------------------------------------------

def small_pool(pool, n=6):
    """The first n ranking prompts, enough to exercise the writer."""
    return {**pool, "prompts": [p for p in pool["prompts"] if p["split"] == "ranking"][:n]}


def test_the_writer_writes_a_record_with_everything_needed_to_replay_it(pool, tmp_path):
    sp = small_pool(pool, 1)
    fake, s = FakeOllama(), FakeSafety()
    assert dp.write("ranking", fake.client(), s, sp, tmp_path, out=lambda *_: None) == 1
    rec = dp._read(dp.bit_path(tmp_path, "ranking", sp["prompts"][0]["id"]))
    assert len(rec["threads"]) == 3 and rec["generator_id"].startswith(generate.PROMPT_ID)
    assert rec["request"] and rec["digest"] and rec["wall_s"] >= 0 and rec["safety_rules"] == safety.RULES_VERSION
    assert {"mode", "context", "bit", "told", "day", "gist", "lost", "failure", "bad_replies", "written_at"} <= set(rec)
    assert len(fake.seen) == 3  # one bit, three parallel calls, nothing more


def test_the_writer_resumes_and_never_rewrites_a_finished_bit(pool, tmp_path):
    sp = small_pool(pool)
    fake, s = FakeOllama(), FakeSafety()
    assert dp.write("ranking", fake.client(), s, sp, tmp_path, n=2, out=lambda *_: None) == 2
    first = {p["id"]: dp._read(dp.bit_path(tmp_path, "ranking", p["id"])) for p in sp["prompts"][:2]}
    assert dp.write("ranking", fake.client(), s, sp, tmp_path, out=lambda *_: None) == 4
    assert len(fake.seen) == 3 * 6
    assert all(dp._read(dp.bit_path(tmp_path, "ranking", i)) == r for i, r in first.items())
    assert dp.write("ranking", fake.client(), s, sp, tmp_path, out=lambda *_: None) == 0 and len(fake.seen) == 18


def test_ctrl_c_mid_bit_leaves_only_whole_files_and_the_rerun_finishes(pool, tmp_path, monkeypatch):
    sp = small_pool(pool, 4)
    real, calls = dp.write_bit, []

    def interrupted(*a, **k):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return real(*a, **k)
    monkeypatch.setattr(dp, "write_bit", interrupted)
    with pytest.raises(KeyboardInterrupt):
        dp.write("ranking", FakeOllama().client(), FakeSafety(), sp, tmp_path, out=lambda *_: None)
    files = sorted((tmp_path / "bits" / "ranking").iterdir())
    assert len(files) == 2 and all(f.suffix == ".json" and json.loads(f.read_text()) for f in files)  # no .tmp, no half file
    monkeypatch.setattr(dp, "write_bit", real)
    assert dp.write("ranking", FakeOllama().client(), FakeSafety(), sp, tmp_path, out=lambda *_: None) == 2


def test_a_thread_the_safety_check_rejects_is_recorded_and_kept_out_of_pairs(pool, tmp_path):
    sp = small_pool(pool, 1)
    fake = FakeOllama(bad={1})
    dp.write("ranking", fake.client(), FakeSafety(), sp, tmp_path, out=lambda *_: None)
    rec = dp._read(dp.bit_path(tmp_path, "ranking", sp["prompts"][0]["id"]))
    bad = [t for t in rec["threads"] if not t["safe"]]
    assert len(bad) == 1 and bad[0]["rejected"] == ["rule:test"]
    left, right = dp.pair_for(rec)
    assert rec["threads"][left]["safe"] and rec["threads"][right]["safe"] and left != right


def test_fewer_than_two_clean_threads_gives_no_pair():
    t = lambda ok: {"premise": "p", "lines": ["x"] * 5, "safe": ok}  # noqa: E731
    assert dp.pair_for({"id": "a", "threads": [t(True), t(False), t(False)]}) is None
    assert dp.pair_for({"id": "a", "threads": []}) is None and dp.pair_for(None) is None


def test_a_bit_with_no_thread_is_recorded_as_a_failure_not_a_crash(pool, tmp_path):
    def refuse(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}, "finish_reason": "stop"}]})
    client = llm.Client("http://local/v1", "", "bonsai-2:4b", 10.0, local=True, transport=httpx.MockTransport(refuse),
                        sleep=lambda s: None)
    sp = small_pool(pool, 1)
    assert dp.write("ranking", client, FakeSafety(), sp, tmp_path, out=lambda *_: None) == 1
    rec = dp._read(dp.bit_path(tmp_path, "ranking", sp["prompts"][0]["id"]))
    assert rec["threads"] == [] and rec["failure"] and rec["lost"] and not dp.done(rec)
    # the next run writes it again rather than counting it as finished
    fake = FakeOllama()
    assert dp.write("ranking", fake.client(), FakeSafety(), sp, tmp_path, out=lambda *_: None) == 1 and len(fake.seen) == 3
    assert dp.done(dp._read(dp.bit_path(tmp_path, "ranking", sp["prompts"][0]["id"])))


def test_the_writer_stops_when_bits_keep_coming_back_empty(pool, tmp_path):
    def down(request):
        raise httpx.ConnectError("refused")
    client = llm.Client("http://local/v1", "", "bonsai-2:4b", 10.0, local=True, transport=httpx.MockTransport(down),
                        sleep=lambda s: None)
    sp = small_pool(pool, 6)
    with pytest.raises(SystemExit, match="3 bits in a row"):
        dp.write("ranking", client, FakeSafety(), sp, tmp_path, out=lambda *_: None)
    assert len(list((tmp_path / "bits" / "ranking").iterdir())) == 3  # recorded, and retried on the next run


def test_a_record_written_for_another_prompt_with_the_same_id_is_written_again(pool, tmp_path):
    sp = small_pool(pool, 1)
    p = sp["prompts"][0]
    dp.write("ranking", FakeOllama().client(), FakeSafety(), sp, tmp_path, out=lambda *_: None)
    moved = {**sp, "prompts": [{**p, "context": {**p["context"], "venue": "elsewhere"}}]}
    fake = FakeOllama()
    assert dp.write("ranking", fake.client(), FakeSafety(), moved, tmp_path, out=lambda *_: None) == 1 and len(fake.seen) == 3
    assert dp._read(dp.bit_path(tmp_path, "ranking", p["id"]))["context"]["venue"] == "elsewhere"


def test_when_only_the_safety_layer_failed_the_rerun_rechecks_without_regenerating(pool, tmp_path):
    sp = small_pool(pool, 1)
    fake = FakeOllama()
    dp.write("ranking", fake.client(), FakeSafety(fail=True), sp, tmp_path, out=lambda *_: None)
    path = dp.bit_path(tmp_path, "ranking", sp["prompts"][0]["id"])
    first = dp._read(path)
    assert first["safety_infra_error"] and not any(t["safe"] for t in first["threads"])
    assert dp.write("ranking", fake.client(), FakeSafety(), sp, tmp_path, out=lambda *_: None) == 1
    again = dp._read(path)
    assert not again["safety_infra_error"] and all(t["safe"] for t in again["threads"])
    assert [t["lines"] for t in again["threads"]] == [t["lines"] for t in first["threads"]] and len(fake.seen) == 3


def test_the_tonight_half_sets_the_days_allow_lists_on_the_safety_check(pool, tmp_path):
    prompt = next(p for p in pool["prompts"] if p["half"] == "tonight" and p["bit"] == 3)
    d = day(people=["Jane Roe"])
    d["names"] = ["Thanksgiving"]
    s = FakeSafety()
    dp.write_bit(FakeOllama().client(), s, prompt, {prompt["day"]: d})
    assert s.allow_people == frozenset({"Jane Roe"}) and "Thanksgiving" in s.allow_caps
    # one safety check serves the whole split: a plain bit after a tonight bit allows nothing again
    plain = next(p for p in pool["prompts"] if p["half"] == "plain")
    dp.write_bit(FakeOllama().client(), s, plain, {prompt["day"]: d})
    assert s.allow_people == frozenset() and s.allow_caps == frozenset()


def test_a_tonight_bit_rechecked_after_a_safety_outage_gets_its_days_allow_lists(pool, tmp_path):
    d = day(people=["Jane Roe"])
    prompts = [next(p for p in pool["prompts"] if p["split"] == "ranking" and p["half"] == h) for h in ("tonight", "plain")]
    sp = {**pool, "days": [d], "prompts": prompts}
    dp.write("ranking", FakeOllama().client(), FakeSafety(fail=True), sp, tmp_path, out=lambda *_: None)
    seen = []

    class Watch(FakeSafety):
        def check(self, text, age):
            seen.append(self.allow_people)
            return super().check(text, age)
    dp.write("ranking", FakeOllama().client(), Watch(), sp, tmp_path, out=lambda *_: None)
    assert seen[:15] == [frozenset({"Jane Roe"})] * 15 and seen[15:] == [frozenset()] * 15


# --- the ranking tool ----------------------------------------------------------------------------------------------------

def sitting(pool, tmp_path, answers, limit=None):
    out = []
    dp.rank(pool, tmp_path, limit=limit, read=scripted(*answers), out=out.append)
    return out


def test_ties_and_skips_are_replaced_so_120_decided_pairs_split_30_in_every_cell(pool, tmp_path):
    fake_bits(pool, tmp_path)
    cycle = list(itertools.islice(itertools.cycle(["1", "2", "t", "1", "s", "2"]), 400))
    out = sitting(pool, tmp_path, cycle)
    state = dp.read_state(tmp_path)
    assert dp.decided(state, pool) == {c: 30 for c in dp.CELLS}
    assert sum(a["choice"] in "12" for a in state["answers"].values()) == 120
    assert out[-1].startswith("120 of 120 done (ties ") and "teach" not in out[-1] and "test" not in out[-1]
    ties = sum(a["choice"] == "t" for a in state["answers"].values())
    skips = sum(a["choice"] == "s" for a in state["answers"].values())
    assert ties > 0 and skips > 0 and f"ties {ties}, skips {skips})" in out[-1]
    pairs = dp.decided_pairs(pool, state, tmp_path)
    assert len(pairs) == 120 and collections.Counter((p["role"], p["half"]) for p in pairs) == {c: 30 for c in dp.CELLS}
    assert not any(a["choice"] in "ts" and a["choice"] == "x" for a in state["answers"].values())


def test_a_tie_or_skip_never_comes_back(pool, tmp_path):
    fake_bits(pool, tmp_path)
    sitting(pool, tmp_path, ["t", "s", "t", "s"])
    state = dp.read_state(tmp_path)
    assert len(state["answers"]) == 4 and dp.decided(state, pool) == {c: 0 for c in dp.CELLS}
    sitting(pool, tmp_path, ["1"] * 6)
    assert len(dp.read_state(tmp_path)["answers"]) == 10  # six new prompts, none of the four again


def test_a_sitting_is_resumable_limited_and_saved_after_every_answer(pool, tmp_path):
    fake_bits(pool, tmp_path)
    out = sitting(pool, tmp_path, ["1"] * 50, limit=20)
    assert out[-1].startswith("20 of 120 done") and len(dp.read_state(tmp_path)["answers"]) == 20
    # a stop in the middle of a sitting keeps what was answered
    out = sitting(pool, tmp_path, ["2", "1", "q"])
    assert len(dp.read_state(tmp_path)["answers"]) == 22 and out[-1].startswith("22 of 120 done")
    # the balance holds in a short sitting: 20 questions are 5 per cell
    first20 = dp.read_state(tmp_path)["answers"]
    roles = {p["id"]: (p["role"], p["half"]) for p in pool["prompts"] if p["split"] == "ranking"}
    assert collections.Counter(roles[i] for i in list(first20)[:20]) == {c: 5 for c in dp.CELLS}


def test_the_order_of_questions_does_not_give_away_which_are_test_pairs(pool, tmp_path):
    fake_bits(pool, tmp_path)
    sitting(pool, tmp_path, ["1"] * 40)
    roles = {p["id"]: (p["role"], p["half"]) for p in pool["prompts"] if p["split"] == "ranking"}
    cells = [roles[i] for i in dp.read_state(tmp_path)["answers"]]
    blocks = [cells[k:k + 4] for k in range(0, 40, 4)]
    assert all(sorted(b) == sorted(dp.CELLS) for b in blocks)  # still balanced: one of each cell every 4 questions
    assert len({tuple(b) for b in blocks}) > 1  # but not in a fixed order, so position says nothing about the role


def test_teach_and_test_are_fixed_before_any_answer_and_do_not_depend_on_answers(pool, tmp_path):
    fake_bits(pool, tmp_path)
    before = {p["id"]: p["role"] for p in pool["prompts"]}
    sitting(pool, tmp_path, ["1", "2", "t", "s"] * 30)
    assert {p["id"]: p["role"] for p in pool["prompts"]} == before
    state = dp.read_state(tmp_path)
    assert state["seed"] == dp.RANK_SEED
    roles = {p["id"]: p["role"] for p in pool["prompts"]}
    assert all(dp.decided_pairs(pool, state, tmp_path)[i]["role"] == roles[p["id"]]
               for i, p in enumerate(p for p in pool["prompts"] if p["id"] in state["answers"] and state["answers"][p["id"]]["choice"] in "12"))


def test_the_user_sees_two_clean_threads_of_one_bit_five_lines_each_and_no_labels(pool, tmp_path):
    fake_bits(pool, tmp_path)
    out = sitting(pool, tmp_path, ["1"], limit=1)
    shown = "\n".join(out[:-1])  # the last line is the progress count, which is fine to see
    assert shown.count(" line ") == 10 and "--- 1" in shown and "--- 2" in shown
    assert "tonight" not in shown and "teach" not in shown and "test" not in shown


def test_the_ranking_waits_for_the_writer_when_no_bit_is_ready(pool, tmp_path):
    out = sitting(pool, tmp_path, ["1"])
    assert out[0].startswith("No pair ready") and out[-1].startswith("0 of 120 done")


def test_a_bit_with_one_clean_thread_is_skipped_not_shown(pool, tmp_path):
    fake_bits(pool, tmp_path)
    first = next(p for p in pool["prompts"] if p["split"] == "ranking" and (p["role"], p["half"]) == ("teach", "tonight"))
    rec = dp._read(dp.bit_path(tmp_path, "ranking", first["id"]))
    for t in rec["threads"][1:]:
        t["safe"] = False
    dp._write(dp.bit_path(tmp_path, "ranking", first["id"]), rec)
    sitting(pool, tmp_path, ["1"] * 5)
    assert first["id"] not in dp.read_state(tmp_path)["answers"]


# --- the gate ------------------------------------------------------------------------------------------------------------

def fake_pairs(n_test_right_first=60):
    """60 teach and 60 test pairs, half and half by tonight half. Every winning thread's lines carry WIN."""
    pairs = []
    for role in dp.ROLES:
        for k in range(60):
            half = dp.HALVES[k % 2]
            winner = k % 2
            lines = lambda tag: [f"{role} {k} {tag} line {j}." for j in range(5)]  # noqa: E731
            sides = [lines("WIN" if winner == 0 else "lose"), lines("WIN" if winner == 1 else "lose")]
            pairs.append({"id": f"{role}{k:02d}", "role": role, "half": half, "left": sides[0], "right": sides[1], "winner": winner})
    return pairs


class FakeJudge:
    """Pairs numbered below `right` get the bit with the WIN marker, pairs from `right` up to `consistent` get the other
    bit (both consistent in the two orders); pairs from `consistent` up always get A, so the two orders disagree.
    `positional` makes every pair do that; a pair numbered in `down` fails (an outage)."""

    def __init__(self, right=60, consistent=60, positional=False, down=()):
        self.right, self.consistent, self.positional, self.down, self.calls = right, consistent, positional, set(down), []

    def chat(self, system, user, *, json_reply=False, temperature=0.0, validate=None, deadline_s=None):
        self.calls.append((system, user))
        a_block = user.split("Bit B:")[0]
        pair_no = int(a_block.split("test ")[1].split(" ")[0])
        if pair_no in self.down:
            raise llm.LLMError("down")
        if self.positional or pair_no >= self.consistent:
            return json.dumps({"funnier": "A"})
        return json.dumps({"funnier": "A" if ("WIN" in a_block) == (pair_no < self.right) else "B"})


def judged(tmp_path, pairs, **kw):
    judge = FakeJudge(**kw)
    dp.judge_all(judge, pairs, tmp_path, out=lambda *_: None)
    return judge, dp.summarize(pairs, dp.read_letters(pairs, tmp_path))


def test_the_judge_is_asked_each_test_pair_in_both_orders_and_no_teach_pair(tmp_path):
    pairs = fake_pairs()
    judge, res = judged(tmp_path, pairs)
    assert len(judge.calls) == 120 and res["n_test"] == 60
    users = [u for _, u in judge.calls]
    for p in (p for p in pairs if p["role"] == "test"):
        mine = [u for u in users if f"test {int(p['id'][4:])} " in u]
        assert len(mine) == 2 and mine[0] != mine[1]  # the same two threads, swapped
        assert dp.judge_user(p, "ab") in users and dp.judge_user(p, "ba") in users
    assert len({s for s, _ in judge.calls}) == 1  # one teach prompt for every call


def test_test_pairs_are_never_in_the_teach_prompt_and_every_teach_pair_is(tmp_path):
    pairs = fake_pairs()
    system = dp.teach_prompt(pairs)
    assert system.count("Their pick:") == 60 and "Pair 60" in system
    for p in pairs:
        text = dp.lines_text(p["left"])
        assert (text in system) == (p["role"] == "teach")
    # a test pair smuggled into the teach role is the one thing the guard cannot see: roles are the contract
    leaked = [dict(p) for p in pairs]
    leaked.append({**leaked[-1], "role": "test", "id": "test99"})
    leaked[-1]["left"] = leaked[0]["left"]  # identical lines to a teach pair
    with pytest.raises(AssertionError, match="test pair is in the teach prompt"):
        dp.teach_prompt(leaked)


def test_a_judge_that_always_agrees_passes_with_no_flips(tmp_path):
    _, res = judged(tmp_path, fake_pairs())
    assert (res["verdict"], res["agree"], res["consistent"], res["flip_rate"], res["agree_single_order"]) == ("pass", 60, 60, 0.0, 1.0)
    assert res["by_half"]["tonight"]["agree"] == 30 and res["by_half"]["plain"]["agree"] == 30


# c = pairs answered the same way in both orders, a = those agreeing with you. The minimum a that clears the Wilson
# lower bound above one half is 21 of 30, 38 of 60 (computed with jc.wilson).
@pytest.mark.parametrize("c,a,verdict", [(29, 29, "fail"), (30, 21, "pass"), (30, 20, "fail"), (30, 30, "pass"),
                                         (45, 30, "pass"), (60, 38, "pass"), (60, 37, "fail"), (60, 60, "pass"), (60, 0, "fail"),
                                         (0, 0, "fail")])
def test_the_gate_needs_30_consistent_pairs_and_a_lower_bound_above_half_among_them(tmp_path, c, a, verdict):
    _, res = judged(tmp_path, fake_pairs(), right=a, consistent=c)
    assert (res["consistent"], res["agree_consistent"], res["verdict"]) == (c, a, verdict)
    assert res["agree"] == a  # the strict count over all 60 is the same a: pairs without a label never agree


def test_the_minimum_agreeing_pairs_are_the_computed_ones():
    mins = {c: next(a for a in range(c + 1) if jc.wilson(a, c)[0] > 0.5) for c in (30, 45, 60)}
    assert mins == {30: 21, 45: 30, 60: 38}
    assert all(dp.passes(mins[c], c) and not dp.passes(mins[c] - 1, c) for c in mins)
    assert not dp.passes(29, 29) and dp.MIN_CONSISTENT == 30  # coverage fails first, whatever the agreement


def test_a_positional_judge_flips_and_gets_no_label(tmp_path):
    _, res = judged(tmp_path, fake_pairs(), positional=True)
    assert res["agree"] == 0 and res["consistent"] == 0 and res["flip_rate"] == 1.0 and res["verdict"] == "fail"
    assert res["agree_single_order"] == 0.5  # always A is right in half of the orders: position, not taste


def test_a_missing_call_makes_the_gate_incomplete_not_a_fail(tmp_path):
    class Down(FakeJudge):
        def chat(self, system, user, **kw):
            if "test 7 " in user:
                raise llm.LLMError("down")
            return super().chat(system, user, **kw)
    pairs = fake_pairs()
    dp.judge_all(Down(), pairs, tmp_path, out=lambda *_: None)
    res = dp.summarize(pairs, dp.read_letters(pairs, tmp_path))
    assert res["verdict"] == "incomplete" and res["missing_pairs"] == 1
    # the rerun asks only what is missing
    again = FakeJudge()
    dp.judge_all(again, pairs, tmp_path, out=lambda *_: None)
    assert len(again.calls) == 2 and dp.summarize(pairs, dp.read_letters(pairs, tmp_path))["verdict"] == "pass"


def test_a_pair_whose_two_orders_disagree_gets_no_label_and_is_not_counted_either_way():
    pairs = [p for p in fake_pairs() if p["role"] == "test"]
    letters = {}
    for p in pairs:
        w = p["winner"]
        letters[(p["id"], "ab")] = "A" if w == 0 else "B"  # right in the first order
        letters[(p["id"], "ba")] = "A" if w == 0 else "B"  # same letter in the swapped order: the other thread
    pairs += [p for p in fake_pairs() if p["role"] == "teach"]
    res = dp.summarize(pairs, letters)
    assert res["agree"] == 0 and res["consistent"] == 0 and res["flip_rate"] == 1.0 and res["agree_single_order"] == 0.5
    assert res["agree_flip_half_credit"] == 0.5  # D-95's convention, reported beside the bar and never deciding


def test_the_judge_cache_follows_the_teach_prompt(tmp_path):
    pairs = fake_pairs()
    judged(tmp_path, pairs)
    changed = [dict(p) for p in pairs]
    changed[0]["winner"] = 1 - changed[0]["winner"]  # a teach answer changed
    assert dp.judge_dir(tmp_path, dp.teach_prompt(changed)) != dp.judge_dir(tmp_path, dp.teach_prompt(pairs))
    assert dp.read_letters(changed, tmp_path)[("test00", "ab")] is None  # nothing is reused for a different teach prompt


def test_the_record_key_changes_when_any_answer_changes(pool):
    state = {"seed": dp.RANK_SEED, "answers": {"r0001": {"choice": "1", "left": 0, "right": 1}}}
    pairs = fake_pairs()
    key = dp.params(pool, state, pairs)
    flipped = {**state, "answers": {"r0001": {"choice": "2", "left": 0, "right": 1}}}
    assert dp.params(pool, flipped, pairs)["answers_digest"] != key["answers_digest"]
    assert set(key) == set(dp.run.__code__.co_varnames[:dp.run.__code__.co_argcount])


def test_the_record_stores_numbers_only(tmp_path, monkeypatch, pool):
    monkeypatch.setattr(dp.cache, "METRICS", tmp_path)
    _, res = judged(tmp_path, fake_pairs())
    saved = dp.cache.run_or_load(dp.NAME, lambda **_: res, rerun=True, pool_digest="x", rank_seed=dp.RANK_SEED,
                                 answers_digest="z", judge_digest_="y", min_consistent=dp.MIN_CONSISTENT)
    assert saved == res
    text = (tmp_path / f"{dp.NAME}.json").read_text()
    assert "line" not in text and "WIN" not in text
    assert json.loads(text)["results"]["verdict"] == "pass"


# --- rebuilding only the training split ----------------------------------------------------------------------------------

def put_fetch(directory, date, n_gists=3):
    gists = [f"{date} gist {i}" for i in range(n_gists)]
    dp._write(directory / date / "fetch.json", {"date": date, "fetched_at": date, "approved": gists, "market": {"line": f"up on {date}"},
                                                  "facts": [], "keep": {}, "people": {}})


def frozen(tmp_path):
    fetches, root = tmp_path / "tonight", tmp_path / "dpo"
    put_fetch(fetches, "2026-11-26")
    pool = dp.build_pool(dp.load_days(fetches))
    dp._write(dp.pool_path(root), pool)
    return fetches, root, pool


def test_rebuilding_the_training_split_keeps_the_ranking_split_and_stays_disjoint(tmp_path):
    fetches, root, old = frozen(tmp_path)
    put_fetch(fetches, "2026-11-27")
    put_fetch(fetches, "2026-11-28")
    new = dp.rebuild_train(root, fetches)
    pick = lambda pool, split: [p for p in pool["prompts"] if p["split"] == split]  # noqa: E731
    assert pick(new, "ranking") == pick(old, "ranking")  # roles, gists and rooms are untouched
    assert pick(new, "train") != pick(old, "train") and len(pick(new, "train")) == dp.TRAIN_N
    assert [d["date"] for d in new["days"]] == ["2026-11-26", "2026-11-27", "2026-11-28"]
    assert {p["day"] for p in pick(new, "train") if p["gist"]} == {"2026-11-26", "2026-11-27", "2026-11-28"}
    dp.assert_disjoint(new)  # still disjoint from the ranking split and Exp 24, 26, 27, 30
    assert dp._read(dp.pool_path(root)) == new


def test_old_training_bits_are_moved_aside_not_deleted_on_a_rebuild(tmp_path):
    fetches, root, old = frozen(tmp_path)
    dp._write(dp.bit_path(root, "train", "t0000"), {"id": "t0000"})
    put_fetch(fetches, "2026-11-27")
    dp.rebuild_train(root, fetches)
    assert not (root / "bits" / "train").exists()
    assert dp._read(root / "bits" / f"train.before-{old['digest']}" / "t0000.json") == {"id": "t0000"}


def test_a_rebuild_keeps_the_bits_of_prompts_it_left_unchanged(tmp_path):
    fetches, root, old = frozen(tmp_path)
    train = [p for p in old["prompts"] if p["split"] == "train"]
    plain = next(p for p in train if p["half"] == "plain")
    tonight = next(p for p in train if p["half"] == "tonight" and p["gist"])
    for p in (plain, tonight):
        dp._write(dp.bit_path(root, "train", p["id"]), {**{k: p[k] for k in dp.PROMPT_FIELDS}, "id": p["id"], "threads": []})
    put_fetch(fetches, "2026-11-27")
    new = dp.rebuild_train(root, fetches)
    assert next(p for p in new["prompts"] if p["id"] == plain["id"]) == plain
    assert dp._read(dp.bit_path(root, "train", plain["id"]))["id"] == plain["id"]  # the plain half is unchanged: kept
    assert next(p for p in new["prompts"] if p["id"] == tonight["id"]) != tonight
    assert dp._read(dp.bit_path(root, "train", tonight["id"])) is None  # its gist moved with the new days: set aside
    # a second rebuild never overwrites the first one's set-aside bits
    dp._write(dp.bit_path(root, "train", "t9999"), {"id": "t9999"})
    put_fetch(fetches, "2026-11-28")
    dp.rebuild_train(root, fetches)
    assert len([d for d in (root / "bits").iterdir() if d.name.startswith("train.before-")]) == 2


def test_a_labelled_training_split_cannot_be_rebuilt(tmp_path):
    fetches, root, old = frozen(tmp_path)
    dp._write(root / "labels" / "t0000.json", {"label": 1})
    put_fetch(fetches, "2026-11-27")
    with pytest.raises(SystemExit, match="already labelled"):
        dp.rebuild_train(root, fetches)
    assert dp._read(dp.pool_path(root)) == old


def test_a_rebuild_refuses_when_a_day_the_ranking_split_used_is_gone_or_changed(tmp_path):
    fetches, root, old = frozen(tmp_path)
    dp._write(fetches / "2026-11-26" / "fetch.json", {"date": "2026-11-26", "fetched_at": "later", "approved": ["other"],
                                                       "market": {"line": "changed"}, "facts": [], "keep": {}, "people": {}})
    with pytest.raises(SystemExit, match="missing or changed"):
        dp.rebuild_train(root, fetches)
    assert dp._read(dp.pool_path(root)) == old
