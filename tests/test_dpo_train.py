"""D-123: the writer flag and per-writer folders, the Haiku labeller, the dataset export and the Modelfile, with a fake Ollama and a
fake judge. No model, no network."""
import dataclasses
import json
import pathlib
import re
import sys

import httpx
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app import generate, llm  # noqa: E402
from app import tonight as app_tonight  # noqa: E402
from sim import dpo_pairs as dp  # noqa: E402
from sim import dpo_train as dt  # noqa: E402
from tests.test_dpo_pairs import FakeOllama, FakeSafety, day  # noqa: E402

QWEN = "qwen3:4b-instruct"


def client_for(model, fake):
    return llm.Client("http://local/v1", "", model, 10.0, retries=2, deadline_s=12.0, local=True, extra_body=llm.OLLAMA_NO_THINK,
                      json_schema=True, transport=httpx.MockTransport(fake))


@pytest.fixture(scope="module")
def pool():
    return dp.build_pool([day()])


# --- the writer flag -----------------------------------------------------------------------------------------------------

def test_writer_client_takes_a_model_and_changes_nothing_else(monkeypatch):
    monkeypatch.setenv("JOKE_WRITER_PROVIDER", "ollama")
    monkeypatch.delenv("JOKE_WRITER_MODEL", raising=False)
    live, other = generate.writer_client(), generate.writer_client(QWEN)
    assert (live.model, other.model) == (llm.OLLAMA_MODEL, QWEN)
    assert dataclasses.replace(live, model=QWEN) == other


def test_each_writer_gets_its_own_folder_and_the_first_keeps_the_old_one(tmp_path):
    assert dp.bits_folder("train", llm.OLLAMA_MODEL) == "train" and dp.bits_folder("train") == "train"
    assert dp.bits_folder("train", QWEN) == "train-qwen3"
    assert dp.bits_folder("eval", "humor-qwen3-B") == "eval-humor-qwen3-B"
    assert dp.bit_path(tmp_path, "train", "t0001", QWEN) == tmp_path / "bits" / "train-qwen3" / "t0001.json"
    assert dp.bit_path(tmp_path, "ranking", "r0001") == tmp_path / "bits" / "ranking" / "r0001.json"


def test_the_request_is_the_shows_for_every_writer(pool, monkeypatch):
    """The body sent for a pool prompt with writer X is what a show sends with X: only the model name differs between writers."""
    d = day()
    prompt = next(p for p in pool["prompts"] if p["split"] == "train" and p["half"] == "tonight" and p["bit"] == 2)
    bodies = {}
    for model in (llm.OLLAMA_MODEL, QWEN):
        fake = FakeOllama()
        gen = dp.new_generator(client_for(model, fake), prompt, {d["date"]: d})
        gen.generate(prompt["mode"], prompt["bit"], 0, prompt["context"])
        show = FakeOllama()
        sg = generate.ThreadGenerator(client_for(model, show), told=prompt["told"])
        sg.set_tonight(app_tonight.TonightContext(d["facts"], d["market"], [prompt["gist"]] + [g for g in d["approved"] if g != prompt["gist"]],
                                                  allowed_names=d["names"], allowed_people=d["people"]))
        sg.generate(prompt["mode"], prompt["bit"], 0, prompt["context"])
        assert fake.seen[0] == show.seen[0] and gen.generator_id == sg.generator_id
        bodies[model] = fake.seen[0]
    a, b = (dict(bodies[m]) for m in (llm.OLLAMA_MODEL, QWEN))
    assert (a.pop("model"), b.pop("model")) == (llm.OLLAMA_MODEL, QWEN)
    assert a == b  # system, user, schema, temperature and the no-thinking switch are the same
    assert b["temperature"] == generate.TEMPERATURE and b["reasoning_effort"] == "none"
    assert b["response_format"]["json_schema"]["schema"] == generate.COUNTED_REPLY_SCHEMA


def train_prompts(pool, n):
    return [p for p in pool["prompts"] if p["split"] == "train"][:n]


def test_write_stores_the_writer_the_raw_reply_and_a_digest_of_its_own(pool, tmp_path):
    fake = FakeOllama()
    assert dp.write("train", client_for(QWEN, fake), FakeSafety(), pool, tmp_path, n=2, out=lambda *_: None) == 2
    first = train_prompts(pool, 1)[0]
    rec = dp._read(tmp_path / "bits" / "train-qwen3" / f"{first['id']}.json")
    assert rec["writer"] == QWEN and rec["generator_id"].endswith(f":{QWEN}") and not (tmp_path / "bits" / "train").exists()
    assert all(t["raw"] == json.dumps({"premise": t["premise"], "jokes": t["lines"]}) for t in rec["threads"])
    other = FakeOllama()
    dp.write("train", client_for(llm.OLLAMA_MODEL, other), FakeSafety(), pool, tmp_path, n=1, out=lambda *_: None)
    old = dp._read(tmp_path / "bits" / "train" / f"{first['id']}.json")
    assert old["writer"] == llm.OLLAMA_MODEL and old["digest"] != rec["digest"]  # the same request, another writer


def test_a_bit_written_by_another_writer_is_not_taken_for_this_ones(pool, tmp_path):
    p = train_prompts(pool, 1)[0]
    dp.write("train", client_for(QWEN, FakeOllama()), FakeSafety(), pool, tmp_path, n=1, out=lambda *_: None)
    rec = dp._read(dp.bit_path(tmp_path, "train", p["id"], QWEN))
    assert dp.same_prompt(rec, p, QWEN) and not dp.same_prompt(rec, p, llm.OLLAMA_MODEL) and dp.same_prompt(rec, p)
    legacy = {k: v for k, v in rec.items() if k != "writer"}  # written before D-123: it is the first writer's
    assert dp.same_prompt(legacy, p, llm.OLLAMA_MODEL) and not dp.same_prompt(legacy, p, QWEN)
    # copied into the wrong folder by hand, it is still refused
    dp._write(dp.bit_path(tmp_path, "train", p["id"], llm.OLLAMA_MODEL), rec)
    assert dt.written_bits("train", llm.OLLAMA_MODEL, pool, tmp_path) == []


def test_write_resumes_and_top_up_adds_raw_to_new_threads_in_the_writers_folder(pool, tmp_path):
    fake = FakeOllama(bad={1, 2})  # two of the first bit's three calls are rejected by the safety fake
    client, safe = client_for(QWEN, fake), FakeSafety()
    dp.write("train", client, safe, pool, tmp_path, n=1, out=lambda *_: None)
    assert dp.write("train", client, safe, pool, tmp_path, n=1, out=lambda *_: None) == 1  # the next bit, not the same one again
    first = train_prompts(pool, 1)[0]
    rec = dp._read(dp.bit_path(tmp_path, "train", first["id"], QWEN))
    assert dp.clean_count(rec) == 1
    added = dp.topup("train", client, safe, pool, tmp_path, out=lambda *_: None)
    new = dp._read(dp.bit_path(tmp_path, "train", first["id"], QWEN))
    assert added >= 1 and dp.clean_count(new) >= 2 and all("raw" in t for t in new["threads"])


def test_attach_raw_matches_by_parsed_content_and_leaves_a_thread_without_a_reply_alone():
    good = json.dumps({"premise": "p", "jokes": ["a.", "b.", "c.", "d.", "e."]})
    threads = [{"premise": "p", "lines": ["a.", "b.", "c.", "d.", "e."]}, {"premise": "q", "lines": ["a.", "b.", "c.", "d.", "e."]}]
    dp.attach_raw(threads, ["not json", '{"premise": "p", "jokes": ["only one"]}', good])
    assert threads[0]["raw"] == good and "raw" not in threads[1]


# --- the labeller --------------------------------------------------------------------------------------------------------

def teach_pairs():
    return [{"id": f"r{i}", "role": "teach", "half": "plain", "left": [f"teach {i} left {j}." for j in range(5)],
             "right": [f"teach {i} right {j}." for j in range(5)], "winner": i % 2} for i in range(4)]


class FakeJudge:
    """Answers by thread number: the lower-numbered thread is funnier in either order, or always A when `position_bias`."""

    def __init__(self, position_bias=False, fail=False):
        self.calls, self.bias, self.fail = [], position_bias, fail
        self.input_tokens = self.output_tokens = self.cache_read_tokens = self.cache_write_tokens = 0

    def chat(self, system, user, *, json_reply=False, validate=None, **kw):
        self.calls.append(user)
        if self.fail:
            raise llm.LLMError("http 503", 503)
        self.input_tokens, self.output_tokens, self.cache_read_tokens = self.input_tokens + 300, self.output_tokens + 10, self.cache_read_tokens + 18000
        a, b = (int(x) for x in re.findall(r" t(\d) line 0", user))
        letter = "A" if self.bias or a < b else "B"
        return json.dumps({"funnier": letter})


def bit_record(p, n_threads=3, unsafe=(), writer=QWEN, raw=True):
    threads = []
    for i in range(n_threads):
        lines = [f"{p['id']} t{i} line {j}." for j in range(5)]
        t = {"premise": f"premise {i}", "lines": lines, "safe": i not in unsafe, "rejected": []}
        if raw:
            t["raw"] = json.dumps({"premise": t["premise"], "jokes": lines})
        threads.append(t)
    return {**{k: p[k] for k in dp.PROMPT_FIELDS}, "id": p["id"], "writer": writer, "request": f"SYSTEM TEXT\n\nline two\n\n{{\"venue\": \"bar\", \"id\": \"{p['id']}\"}}",
            "threads": threads, "safety_infra_error": False, "generator_id": f"gen-v5:{writer}", "digest": "d"}


@pytest.fixture
def written(pool, tmp_path):
    for k, p in enumerate(train_prompts(pool, 3)):
        dp._write(dp.bit_path(tmp_path, "train", p["id"], QWEN), bit_record(p, n_threads=4, unsafe=(1,) if k == 1 else ()))
    return tmp_path


def test_label_asks_every_pair_in_both_orders_and_drops_the_pairs_the_orders_split(pool, written):
    judge = FakeJudge()
    res = dt.label_all(judge, "train", QWEN, teach_pairs(), pool, written, workers=2, out=lambda *_: None)
    # bits: 4 clean threads (first three used: 3 pairs), 3 clean of 4 with thread 1 unsafe -> threads 0,2,3: 3 pairs, 4 clean: 3 pairs
    assert len(judge.calls) == 3 * 3 * 2 == res["calls"]
    rows = dt.read_jsonl(dt.label_file(written, "train", QWEN))
    assert dt.label_file(written, "train", QWEN).name == "train-qwen3.jsonl"
    assert len(rows) == 9 and all(r["status"] == "label" and r["winner"] == r["i"] for r in rows)
    assert {(r["i"], r["j"]) for r in rows if r["bit"] == train_prompts(pool, 2)[1]["id"]} == {(0, 2), (0, 3), (2, 3)}  # the unsafe thread is out
    biased = FakeJudge(position_bias=True)
    other = written / "other"
    for p in train_prompts(pool, 1):
        dp._write(dp.bit_path(other, "train", p["id"], QWEN), bit_record(p))
    dt.label_all(biased, "train", QWEN, teach_pairs(), pool, other, workers=1, out=lambda *_: None)
    rows = dt.read_jsonl(dt.label_file(other, "train", QWEN))
    assert [r["status"] for r in rows] == ["dropped"] * 3 and all(r["winner"] is None and r["ab"] == r["ba"] == "A" for r in rows)


def test_label_is_resumable_and_a_rerun_makes_no_call(pool, written):
    first = FakeJudge()
    dt.label_all(first, "train", QWEN, teach_pairs(), pool, written, out=lambda *_: None, limit=7)
    assert len(first.calls) == 7
    assert dt.label_summary(dt.read_jsonl(dt.label_file(written, "train", QWEN)))["missing"] > 0
    second = FakeJudge()
    dt.label_all(second, "train", QWEN, teach_pairs(), pool, written, out=lambda *_: None)
    assert len(second.calls) == 18 - 7
    third = FakeJudge()
    dt.label_all(third, "train", QWEN, teach_pairs(), pool, written, out=lambda *_: None)
    assert third.calls == [] and dt.label_summary(dt.read_jsonl(dt.label_file(written, "train", QWEN))) == {
        "pairs": 9, "labelled": 9, "dropped": 0, "missing": 0}


def test_label_cache_is_its_own_and_keyed_by_what_the_judge_was_taught(pool, written):
    dt.label_all(FakeJudge(), "train", QWEN, teach_pairs(), pool, written, out=lambda *_: None)
    caches = list((written / "labels" / "cache").iterdir())
    assert len(caches) == 1 and caches[0].name == dp.judge_digest(dp.teach_prompt(teach_pairs()), "haiku") and not (written / "judge").exists()
    changed = teach_pairs()
    changed[0]["winner"] = 1 - changed[0]["winner"]  # another taught pick: a new digest, so a new cache and fresh answers
    again = FakeJudge()
    dt.label_all(again, "train", QWEN, changed, pool, written, out=lambda *_: None)
    assert len(again.calls) == 18 and len(list((written / "labels" / "cache").iterdir())) == 2


def test_label_prints_the_cost_and_keeps_a_running_total_across_runs(pool, written):
    lines = []
    judge = FakeJudge()
    dt.label_all(judge, "train", QWEN, teach_pairs(), pool, written, out=lines.append, limit=5)
    assert lines[0].startswith("5 calls to make, about $") and any("calls this run, $" in x and "in all for these labels" in x for x in lines)
    first = dt.add_usage(dt.usage_file(written, "train", QWEN), {})
    assert first["input"] == 5 * 300 and first["calls"] == 5
    dt.label_all(FakeJudge(), "train", QWEN, teach_pairs(), pool, written, out=lines.append, limit=4)
    assert dt.add_usage(dt.usage_file(written, "train", QWEN), {})["calls"] == 9
    assert dt.cost_usd({"input": 1_000_000, "output": 1_000_000, "cache_read": 1_000_000, "cache_write": 1_000_000}) == pytest.approx(0.1 + 0.5 + 0.01 + 0.125)


def test_label_stops_after_a_run_of_failures_and_caches_nothing_for_them(pool, written):
    judge = FakeJudge(fail=True)
    lines = []
    dt.label_all(judge, "train", QWEN, teach_pairs(), pool, written, workers=1, out=lines.append)
    assert len(judge.calls) <= dt.FAIL_STOP + 1 and any("failed in a row" in x for x in lines)
    assert not list((written / "labels" / "cache").rglob("*.json"))
    assert all(r["status"] == "missing" for r in dt.read_jsonl(dt.label_file(written, "train", QWEN)))


def test_labelling_freezes_the_training_split(pool, written, monkeypatch):
    dt.label_all(FakeJudge(), "train", QWEN, teach_pairs(), pool, written, out=lambda *_: None)
    dp._write(dp.pool_path(written), pool)
    with pytest.raises(SystemExit, match="already labelled"):
        dp.rebuild_train(written)


def test_a_rebuild_is_refused_once_another_writer_has_written_the_split(pool, written):
    dp._write(dp.pool_path(written), pool)
    with pytest.raises(SystemExit, match="another writer"):
        dp.rebuild_train(written)


# --- the export ----------------------------------------------------------------------------------------------------------

def ranking_bits(pool, root):
    answers = {}
    for k, p in enumerate(q for q in pool["prompts"] if q["split"] == "ranking"):
        dp._write(dp.bit_path(root, "ranking", p["id"]), bit_record(p, writer=llm.OLLAMA_MODEL, raw=False))
        answers[p["id"]] = {"choice": "t" if k % 10 == 9 else "1" if k % 2 == 0 else "2", "left": 0, "right": 1, "at": "x"}
    dp._write(dp.rank_path(root), {"seed": dp.RANK_SEED, "answers": answers})


@pytest.fixture
def exported(pool, written):
    dp._write(dp.pool_path(written), pool)
    ranking_bits(pool, written)
    dt.label_all(FakeJudge(), "train", QWEN, teach_pairs(), pool, written, out=lambda *_: None)
    return written, dt.export(QWEN, written, out=lambda *_: None)


def test_export_b_uses_the_raw_reply_as_the_assistant_text(pool, exported):
    root, manifest = exported
    sft = dt.read_jsonl(root / "train" / "sft_B.jsonl")
    dpo = dt.read_jsonl(root / "train" / "dpo_B.jsonl")
    bit = dp._read(dp.bit_path(root, "train", sft[0]["id"], QWEN))
    assert sft[0]["completion"] == bit["threads"][0]["raw"]  # thread 0 wins most pairs
    assert [m["role"] for m in sft[0]["prompt"]] == ["system", "user"] and sft[0]["prompt"][0]["content"].startswith("SYSTEM TEXT\n\nline two")
    assert sft[0]["prompt"][1]["content"].startswith('{"venue"')
    assert len(sft) == 3 and len(dpo) == 9 and manifest["B"] == {"sft": 3, "dpo": 9, "skipped_no_reply": 0}
    for row in dpo:
        pid, i, j = row["id"].rsplit("-", 2)[0], *map(int, row["id"].rsplit("-", 2)[1:])
        t = dp._read(dp.bit_path(root, "train", pid, QWEN))["threads"]
        assert (row["chosen"], row["rejected"]) == (t[i]["raw"], t[j]["raw"])  # the judge's winner is the lower index


def test_the_sft_rule_picks_the_thread_with_most_wins_then_fewest_losses_then_lowest_index():
    rows = [{"bit": "b", "i": 0, "j": 1, "status": "label", "winner": 1}, {"bit": "b", "i": 0, "j": 2, "status": "label", "winner": 2},
            {"bit": "b", "i": 1, "j": 2, "status": "label", "winner": 1}, {"bit": "x", "i": 0, "j": 1, "status": "dropped", "winner": None}]
    scores = dt.thread_scores(rows)
    assert scores["b"] == {1: [2, 0], 0: [0, 2], 2: [1, 1]}
    assert dt.best_thread(scores["b"]) == 1 and "x" not in scores
    assert dt.best_thread({0: [1, 1], 1: [1, 1], 2: [0, 2]}) == 0
    assert dt.best_thread({0: [1, 0], 1: [1, 1]}) == 0 and dt.best_thread({0: [0, 1]}) is None


def test_a_thread_without_a_stored_reply_is_skipped_and_counted(pool, written):
    p = train_prompts(pool, 1)[0]
    rec = bit_record(p)
    del rec["threads"][0]["raw"]
    rows = [{"bit": p["id"], "i": 0, "j": 1, "status": "label", "winner": 1}, {"bit": p["id"], "i": 1, "j": 2, "status": "label", "winner": 1}]
    sft, dpo, counts = dt.export_b(rows, {p["id"]: rec})
    assert counts == {"sft": 1, "dpo": 1, "skipped_no_reply": 1} and sft[0]["completion"] == rec["threads"][1]["raw"]


def test_export_c_is_the_users_decided_pairs_with_the_prompt_rebuilt_from_the_stored_request(pool, exported):
    root, manifest = exported
    rows = dt.read_jsonl(root / "train" / "dpo_C.jsonl")
    state = dp.read_state(root)
    decided = [pid for pid, a in state["answers"].items() if a["choice"] in ("1", "2")]
    assert len(rows) == len(decided) == 216 and manifest["C"] == {"dpo": 216, "ties_left_out": 24}
    first = rows[0]
    rec = dp._read(dp.bit_path(root, "ranking", first["id"]))
    assert first["prompt"] == dt.split_request(rec["request"])
    win, lose = (0, 1) if state["answers"][first["id"]]["choice"] == "1" else (1, 0)
    assert first["chosen"] == dt.canonical_reply(rec["threads"][win]) and first["rejected"] == dt.canonical_reply(rec["threads"][lose])
    assert "off-policy" in manifest["c_note"]


def test_split_request_inverts_the_generators_join(pool):
    d = day()
    prompt = next(p for p in pool["prompts"] if p["split"] == "train" and p["half"] == "tonight" and p["bit"] == 2)
    fake = FakeOllama()
    gen = dp.new_generator(client_for(QWEN, fake), prompt, {d["date"]: d})
    gen.generate(prompt["mode"], prompt["bit"], 0, prompt["context"])
    sent = fake.seen[0]["messages"]
    request = gen._bits[prompt["bit"]][0]
    assert dt.split_request(request) == sent


def test_the_eval_prompts_are_new_seeded_half_tonight_and_disjoint_from_everything(pool, exported):
    root, manifest = exported
    prompts = dt.read_jsonl(root / "train" / "eval_prompts.jsonl")
    assert len(prompts) == dt.EVAL_N == manifest["eval_prompts"] and sum(p["half"] == "tonight" for p in prompts) == dt.EVAL_N // 2
    assert prompts == dt.build_eval_prompts(pool)  # seeded
    dt.assert_eval_disjoint(prompts, pool)
    assert not {dp.prompt_key(p) for p in prompts} & {dp.prompt_key(p) for p in pool["prompts"]}
    assert not {dp.prompt_key(p) for p in prompts} & dp.earlier_keys()
    assert {p["id"] for p in prompts}.isdisjoint({p["id"] for p in pool["prompts"]}) and all(p["split"] == "eval" for p in prompts)
    bad = json.loads(json.dumps(prompts))
    victim = next(p for p in pool["prompts"] if p["split"] == "train")
    bad[0].update({k: victim[k] for k in ("mode", "context", "told", "half", "day", "gist")})
    with pytest.raises(AssertionError, match="ranking or train"):
        dt.assert_eval_disjoint(bad, pool)


def test_the_manifest_records_counts_and_digests_and_the_files_match_it(pool, exported):
    root, manifest = exported
    assert manifest["pool_digest"] == pool["digest"] and manifest["writer"] == QWEN and manifest["labels"]["labelled"] == 9
    for name, info in manifest["files"].items():
        path = root / "train" / name
        assert info["rows"] == len(dt.read_jsonl(path)) and info["sha8"] == dt.sha(path)
    assert json.loads((root / "train" / "manifest.json").read_text())["sft_rule"] == dt.RULE_SFT


def test_export_refuses_with_no_labels_or_a_pair_still_missing(pool, written):
    dp._write(dp.pool_path(written), pool)
    ranking_bits(pool, written)
    with pytest.raises(SystemExit, match="no labels"):
        dt.export(QWEN, written, out=lambda *_: None)
    dt.label_all(FakeJudge(), "train", QWEN, teach_pairs(), pool, written, out=lambda *_: None, limit=4)
    with pytest.raises(SystemExit, match="not labelled yet"):
        dt.export(QWEN, written, out=lambda *_: None)


# --- the laptop's Modelfile ----------------------------------------------------------------------------------------------

BASE_MODELFILE = '''# Modelfile generated by "ollama show"
FROM C:\\Users\\x\\.ollama\\models\\blobs\\sha256-85e4
TEMPLATE """{{ .System }}"""
PARAMETER repeat_penalty 1
PARAMETER stop <|im_end|>
PARAMETER temperature 0.7
PARAMETER top_k 20
PARAMETER top_p 0.8
'''


def test_the_modelfile_keeps_the_template_and_parameters_and_swaps_the_weights():
    out = dt.modelfile_for(BASE_MODELFILE, "runs/gguf/humor-qwen3-B-Q4_K_M.gguf")
    assert "FROM ./humor-qwen3-B-Q4_K_M.gguf\n" in out and "sha256-85e4" not in out
    assert out.replace("FROM ./humor-qwen3-B-Q4_K_M.gguf", "FROM C:\\Users\\x\\.ollama\\models\\blobs\\sha256-85e4") == BASE_MODELFILE
    with pytest.raises(ValueError, match="no FROM"):
        dt.modelfile_for("PARAMETER top_k 20\n", "x.gguf")


def test_the_command_line_writes_a_modelfile_and_prints_the_power_table(tmp_path, monkeypatch, capsys):
    base = tmp_path / "Modelfile.base"
    base.write_text(BASE_MODELFILE, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["dpo_pairs", "modelfile", "--base", str(base), "--gguf", "x/humor-qwen3-C-Q4_K_M.gguf", "--out", str(tmp_path / "Modelfile.C")])
    dp.main()
    assert (tmp_path / "Modelfile.C").read_text(encoding="utf-8").startswith("# Modelfile generated")
    assert "FROM ./humor-qwen3-C-Q4_K_M.gguf" in (tmp_path / "Modelfile.C").read_text(encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["dpo_pairs", "eval", "power"])
    dp.main()
    assert "at least 10" in capsys.readouterr().out
