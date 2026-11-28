"""Tonight's context (D-119): the parsers, the gist filter, the name check, the day cache, the per-show draw, the generator's
request, the log rows and the websocket payload. Synthetic samples and fakes only: no network, no model."""
import asyncio
import datetime
import json
import pathlib
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from app import cache, generate, safety, tonight  # noqa: E402
from app.director import load_director  # noqa: E402
from app.live import load_sessions  # noqa: E402
from app.log import SessionLog, read_session, validate  # noqa: E402
from app.protocol import ProtocolError, parse  # noqa: E402
from app.rooms import RealClock, SimRoom  # noqa: E402
from app.show import UniformSelector, run_show  # noqa: E402
from tests.test_generate import CTX, FAST, FakePlayer, FakeVoice, Filter, parallel_client, thread_reply  # noqa: E402
from tests.test_replay import START, react, same  # noqa: E402

# Synthetic samples in the formats checked on 2026-11-26 (Google News RSS; FRED's SP500 CSV, which marks a day with no close as ".").
# Stooq, the first choice, answered scripts with an HTML browser check (D-119).
CSV = """observation_date,SP500
2026-11-20,5000.0
2026-11-21,5010.0
2026-11-24,5060.0
2026-11-25,.
"""
RSS = """<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel><title>Top stories</title>
<item><title>Chipmaker unveils faster chip - Example Wire</title><link>https://example.test/1</link></item>
<item><title>Parade postponed for first time - Another Paper</title><link>https://example.test/2</link></item>
<item><title>Parade postponed for first time - Third Outlet</title><link>https://example.test/3</link></item>
<item><title>No source suffix here</title></item>
<item><title>  </title></item>
</channel></rss>"""


# --- parsers ---

def test_market_is_the_last_two_numeric_closes_and_the_line_names_the_date():
    m = tonight.parse_market(CSV)
    assert (m["date"], m["close"], m["prev_close"], m["pct"]) == ("2026-11-24", 5060.0, 5010.0, 1.0)
    assert m["line"] == "The S&P 500 rose 1.0% on Nov 24."
    fell = tonight.parse_market("observation_date,SP500\n2026-12-02,100\n2026-12-03,99.5\n")
    assert fell["line"] == "The S&P 500 fell 0.5% on Dec 3."


def test_the_market_row_order_does_not_matter():
    m = tonight.parse_market(CSV)
    shuffled = "\n".join([CSV.splitlines()[0], *reversed(CSV.splitlines()[1:])])
    assert tonight.parse_market(shuffled) == m


def test_market_line_up_down_and_flat():
    day = datetime.date(2026, 11, 24)
    assert tonight.market_line(1.24, day) == "The S&P 500 rose 1.2% on Nov 24."
    assert tonight.market_line(-0.46, day) == "The S&P 500 fell 0.5% on Nov 24."
    assert "about flat" in tonight.market_line(0.04, day) and "about flat" in tonight.market_line(-0.04, day)


def test_the_source_is_fred_and_no_stooq_is_left():
    assert tonight.MARKET_URL == "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500"
    assert "stooq" not in pathlib.Path(tonight.__file__).read_text(encoding="utf-8").lower()


@pytest.mark.parametrize("text", ["<!DOCTYPE html><html>browser check</html>", "", "observation_date,SP500\n",
                                  "observation_date,SP500\n2026-11-25,.\n2026-11-24,5000\n",
                                  "observation_date,SP500\n2026-11-25,abc\n2026-11-24,5000\n",
                                  "Date,Open,High,Low,Close,Volume\n2026-11-24,1,1,1,1,1\n2026-11-25,1,1,1,1,1\n"])
def test_a_browser_check_page_is_not_a_market_reply(text):
    with pytest.raises(tonight.TonightError):
        tonight.parse_market(text)


def test_headlines_lose_their_source_and_duplicates_and_blanks():
    assert tonight.parse_headlines(RSS) == ["Chipmaker unveils faster chip", "Parade postponed for first time",
                                            "No source suffix here"]
    assert tonight.parse_headlines(RSS, limit=1) == ["Chipmaker unveils faster chip"]


@pytest.mark.parametrize("text", ["not xml", "<rss><channel></channel></rss>"])
def test_a_feed_with_no_items_is_an_error(text):
    with pytest.raises(tonight.TonightError):
        tonight.parse_headlines(text)


# --- the name check ---

@pytest.mark.parametrize("gist", ["A tech CEO says the S&P 500 may fall.", "a GPU company reports earnings", "A senator proposes a bill.",
                                  "The US and the EU talk about AI rules."])
def test_a_nameless_gist_passes_the_name_check(gist):
    assert tonight.name_check(gist) is None


@pytest.mark.parametrize("gist,found", [("A senator meets Priya", "Priya"), ("the mayor opens a park in Springfield", "Springfield"),
                                        ("A chipmaker called NVIDIA grows", "NVIDIA"), ("kevin wins a prize", "kevin"),
                                        ("Rain. Then Tulsa floods", "Tulsa"), ("A chip from Apple's lab", "Apple's")])
def test_a_likely_name_is_found(gist, found):
    assert tonight.name_check(gist) == found


def test_a_capital_after_a_full_stop_starts_a_sentence_and_is_not_a_name():
    assert tonight.name_check("Stocks rose. Analysts shrugged.") is None
    assert tonight.name_check("A U.S. court rules.") is None


# --- the Haiku step ---

def condensed(*items):
    """Items are (category, gist) or (category, gist, [kept names as (name, type)])."""
    return json.dumps({"items": [{"id": i, "category": it[0], "gist": it[1],
                                  "keep": [{"name": n, "type": t} for n, t in (it[2] if len(it) > 2 else [])]}
                                 for i, it in enumerate(items)]})


def test_condensed_reply_must_answer_every_id_once():
    assert tonight.parse_condensed(condensed(("none", " a gist "), ("war", "x")), 2) == [("none", "a gist", [], []), ("war", "x", [], [])]
    for bad in (condensed(("none", "a")), json.dumps({"items": [{"id": 0, "category": "none", "gist": "a"}] * 2}),
                condensed(("sports", "a"), ("none", "b")), json.dumps({"items": [{"id": 0, "category": "none", "gist": 3},
                                                                                {"id": 1, "category": "none", "gist": "b"}]}),
                "not json"):
        with pytest.raises((ValueError, KeyError, TypeError)):
            tonight.parse_condensed(bad, 2)


class FakeClient:
    """Stands in for the Haiku clients: the first call answers `reply`; for the safety check it passes every id."""

    def __init__(self, reply=None, boom=False):
        self.reply, self.boom, self.calls = reply, boom, []

    def chat(self, system, user, *, json_reply=False, temperature=0.0, validate=None, deadline_s=None):
        self.calls.append((system, user))
        if self.boom:
            raise RuntimeError("down")
        if self.reply is not None:
            return self.reply
        n = len(json.loads(user))
        return json.dumps({"verdicts": [{"id": i, "ok": True, "reason": "fine"} for i in range(n)]})


@pytest.fixture(autouse=True)
def fresh_safety_cache():
    safety.clear_cache()
    yield
    safety.clear_cache()


def feeds(market=CSV, news=RSS):
    def get(url, timeout=0):
        body = market if url == tonight.MARKET_URL else news
        if isinstance(body, Exception):
            raise body
        return body
    return get


SIX = ["h0", "h1", "h2", "h3", "h4", "h5"]
SIX_RSS = "<rss><channel>" + "".join(f"<item><title>{h} - Wire</title></item>" for h in SIX) + "</channel></rss>"
SIX_REPLY = condensed(("none", "a tech company unveils a new chip"), ("death", "a famous singer dies"), ("crime", "a man is arrested"),
                      ("none", "a mayor named Kevin opens a park"), ("none", "a senator mocks a fat suit"), ("war", "a city is struck"))


def test_build_counts_every_drop_and_keeps_what_is_left():
    out = tonight.build(feeds(news=SIX_RSS), FakeClient(SIX_REPLY), FakeClient())
    assert out["gists"] == ["a tech company unveils a new chip"]
    assert out["market"]["pct"] == 1.0 and out["errors"] == []
    assert out["report"] == {"fetched": 6, "dropped_by_category": {"death": 1, "disaster": 0, "crime": 1, "war": 1},
                             "dropped_no_gist": 0, "dropped_too_long": 0, "rejected_by_name_check": 1, "rejected_by_safety": 1,
                             "kept": 1}
    assert out["sources"] == {"news": tonight.news_urls(), "market": tonight.MARKET_URL} and out["fetched_at"]
    assert {r["reason"] for r in out["rejected"]} == {"category:death", "category:crime", "category:war", "name:Kevin",
                                                     next(r["reason"] for r in out["rejected"] if r["reason"].startswith("safety:"))}


def test_the_pool_is_top_stories_plus_two_weeks_of_four_sections_deduplicated():
    urls = tonight.news_urls()
    assert urls[0] == tonight.NEWS_URL and len(urls) == 1 + len(tonight.TOPIC_QUERIES)
    assert all("when:14d" in u for u in urls[1:]) and not any(q in " ".join(urls) for q in ("world", "politic"))
    seen = []

    def get(url, timeout=0):
        seen.append(url)
        if url == tonight.MARKET_URL:
            return CSV
        i = urls.index(url)  # each feed has one story of its own and one shared with every other feed
        return f"<rss><channel><item><title>own {i} - Wire</title></item><item><title>shared - Wire</title></item></channel></rss>"
    out = tonight.build(get, FakeClient(condensed(*[("none", f"g{i}") for i in range(6)])), FakeClient())
    assert out["report"]["fetched"] == 6 and set(urls) <= set(seen)


def test_one_dead_feed_costs_only_its_stories():
    urls = tonight.news_urls()

    def get(url, timeout=0):
        if url == urls[2]:
            raise OSError("down")
        return CSV if url == tonight.MARKET_URL else SIX_RSS
    out = tonight.build(get, FakeClient(SIX_REPLY), FakeClient())
    assert out["report"]["fetched"] == 6 and len(out["errors"]) == 1 and urls[2] in out["errors"][0]


def test_a_slow_fetch_is_tried_once_more():
    calls = []

    def get(url, timeout=0):
        calls.append(url)
        if url == tonight.MARKET_URL and calls.count(url) == 1:
            raise TimeoutError("The read operation timed out")
        return CSV if url == tonight.MARKET_URL else SIX_RSS
    out = tonight.build(get, FakeClient(SIX_REPLY), FakeClient())
    assert out["market"] is not None and calls.count(tonight.MARKET_URL) == 2 and out["errors"] == []


def test_months_and_weekdays_pass_the_name_check_but_brands_and_places_do_not():
    assert tonight.name_check("A retailer starts its sale in January, a week before Monday's launch.") is None
    assert tonight.name_check("A search firm called Google cuts prices.") == "Google"
    assert tonight.name_check("Farmers in Tulsa brace for a cold week.") == "Tulsa"


def test_requests_ask_for_text_and_close_the_connection(monkeypatch):
    seen = {}

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"ok"

    def urlopen(req, timeout):
        seen.update(req.header_items())
        return Reply()
    monkeypatch.setattr(tonight.urllib.request, "urlopen", urlopen)
    assert tonight.http_get("https://example.invalid/x") == "ok"
    assert seen["Connection"] == "close" and "text/csv" in seen["Accept"] and seen["Accept-encoding"] == "identity"


def test_many_headlines_are_condensed_in_batches():
    n = tonight.CONDENSE_BATCH + 3
    rss = "<rss><channel>" + "".join(f"<item><title>h{i}</title></item>" for i in range(n)) + "</channel></rss>"

    class Batches(FakeClient):
        def chat(self, system, user, **kw):
            self.calls.append((system, user))
            return condensed(*[("none", f"a gist {len(self.calls)} {i}") for i in range(len(json.loads(user)))])
    client = Batches()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tonight, "PER_FEED", n)
        tonight.build(lambda url, timeout=0: CSV if url == tonight.MARKET_URL else rss, client, FakeClient())
    assert [len(json.loads(u)) for _, u in client.calls] == [tonight.CONDENSE_BATCH, 3]


def test_a_dropped_category_never_reaches_the_safety_check_or_the_operator():
    safety_client = FakeClient()
    tonight.build(feeds(news=SIX_RSS), FakeClient(SIX_REPLY), safety_client)
    sent = " ".join(user for _, user in safety_client.calls)
    assert "singer" not in sent and "arrested" not in sent and "struck" not in sent and "Kevin" not in sent


def test_a_failed_market_still_gives_the_headlines():
    out = tonight.build(feeds(market="<html>check</html>", news=SIX_RSS), FakeClient(SIX_REPLY), FakeClient())
    assert out["market"] is None and len(out["gists"]) == 1 and out["errors"][0].startswith("market:")


@pytest.mark.parametrize("news,client", [(OSError("down"), FakeClient(SIX_REPLY)), (SIX_RSS, FakeClient(boom=True)),
                                         (SIX_RSS, FakeClient("not json"))])
def test_a_failed_headline_half_still_gives_the_market(news, client):
    out = tonight.build(feeds(news=news), client, FakeClient())
    assert out["gists"] == [] and out["market"]["pct"] == 1.0 and out["errors"][0].startswith("headlines:")


def test_a_dead_safety_client_rejects_every_gist():
    out = tonight.build(feeds(news=SIX_RSS), FakeClient(SIX_REPLY), FakeClient(boom=True))
    assert out["gists"] == [] and out["report"]["rejected_by_safety"] == 2  # the chip and the fat-suit one reach it, rules or not


# --- the day cache ---

def test_the_day_is_fetched_once_and_a_refresh_fetches_again(tmp_path):
    calls = []

    def build():
        calls.append(1)
        return {"market": None, "gists": ["g"], "date": "2026-11-26"}
    first = tonight.load_or_build(build, directory=tmp_path, today="2026-11-26")
    assert tonight.load_or_build(build, directory=tmp_path, today="2026-11-26") == first and len(calls) == 1
    tonight.load_or_build(build, refresh=True, directory=tmp_path, today="2026-11-26")
    assert len(calls) == 2
    tonight.load_or_build(build, directory=tmp_path, today="2026-11-27")
    assert len(calls) == 3 and sorted(p.name for p in tmp_path.iterdir()) == ["2026-11-26.json", "2026-11-27.json"]


def test_a_day_where_both_halves_failed_is_not_cached_and_a_damaged_cache_is_a_fetch(tmp_path):
    tonight.load_or_build(lambda: {"market": None, "gists": []}, directory=tmp_path, today="d")
    assert list(tmp_path.iterdir()) == []
    (tmp_path / "d.json").write_text("{broken")
    assert tonight.load_or_build(lambda: {"market": None, "gists": ["g"]}, directory=tmp_path, today="d")["gists"] == ["g"]


def test_the_cache_folder_is_ignored_by_git():
    root = pathlib.Path(__file__).resolve().parents[1]
    assert "data/tonight/" in (root / ".gitignore").read_text()


# --- the per-show context ---

def test_only_bits_2_and_6_draw_a_headline():
    ctx = tonight.TonightContext([], None, ["a", "b", "c"])
    assert [ctx.draw(b) for b in range(8)] == [None, None, "a", None, None, None, "a", None]


def test_a_headline_is_spent_only_when_played_and_only_once_per_bit():
    ctx = tonight.TonightContext([], None, ["a", "b"])
    assert ctx.draw(2) == ctx.draw(2) == "a"  # drawing twice for a bit gives one answer
    assert ctx.use_row(2) is None
    ctx.played(2)
    ctx.played(2)
    assert ctx.use_row(2) == {"gist": "a"} and ctx.draw(6) == "b"
    ctx.played(6)
    assert ctx.draw(6) is None  # out of headlines: the bit goes without a topic


def test_a_bit_that_never_played_leaves_its_headline_for_the_next_headline_bit():
    ctx = tonight.TonightContext([], None, ["a", "b"])
    assert ctx.draw(2) == "a"  # bit 2 fell back to the bank: no played()
    assert ctx.draw(6) == "a"


def test_one_typed_line_at_most_and_nothing_means_no_background():
    assert tonight.MAX_TONIGHT_FACTS == 1
    ctx = tonight.TonightContext(["one", "two"], None, [])
    assert ctx.facts == ["one"] and ctx.background() == {"facts": ["one"], "market": None}
    assert tonight.TonightContext([], None, ["g"]).background() is None
    assert tonight.TonightContext([], None, []).empty()


# --- the approved payload, checked again at show start ---

CACHED = {"market": {"line": "The S&P 500 rose 0.4% on Nov 24."}, "gists": ["a chip appears", "a vote happens", "the Lakers win a game"],
          "keep": {"the Lakers win a game": ["Lakers"]},
          "report": {"fetched": 8, "kept": 2}, "sources": {"news": "n", "market": "m"}, "fetched_at": "t", "date": "d"}


def test_the_payload_becomes_a_context_with_the_fetch_counts_for_the_log():
    payload = {"facts": ["it's snowing"], "market": CACHED["market"]["line"], "gists": ["a vote happens"]}
    ctx = tonight.make_context(payload, CACHED, FakeClient())
    assert (ctx.facts, ctx.market, ctx.gists) == (["it's snowing"], CACHED["market"]["line"], ["a vote happens"])
    row = ctx.summary()
    assert row["report"]["approved"] == 1 and row["report"]["fetched"] == 8 and row["report"]["sources"] == {"news": "n", "market": "m"}
    assert row["report"]["fetched_at"] == "t" and row["report"]["rejected_at_start"] == 0


def test_a_gist_or_market_line_the_fetch_never_made_is_dropped():
    payload = {"facts": [], "market": "The S&P 500 rose 90%.", "gists": ["something the browser invented", "a chip appears"]}
    ctx = tonight.make_context(payload, CACHED, FakeClient())
    assert ctx.gists == ["a chip appears"] and ctx.market is None
    assert tonight.make_context({"facts": [], "market": None, "gists": ["x"]}, None, FakeClient()) is None


def test_a_typed_line_goes_through_the_safety_check_rules_then_model():
    ctx = tonight.make_context({"facts": ["the host has a fat head"], "market": None, "gists": ["a chip appears"]}, CACHED, FakeClient())
    assert ctx.facts == [] and ctx.gists == ["a chip appears"] and ctx.report["rejected_at_start"] == 1
    refusing = FakeClient(json.dumps({"verdicts": [{"id": 0, "ok": False, "reason": "no"}]}))
    assert tonight.make_context({"facts": ["it's a hackathon"], "market": None, "gists": []}, CACHED, refusing) is None


def test_with_no_working_safety_client_nothing_is_approved():
    payload = {"facts": ["it's snowing"], "market": CACHED["market"]["line"], "gists": ["a chip appears"]}
    ctx = tonight.make_context(payload, CACHED, FakeClient(boom=True))
    assert ctx is not None and ctx.facts == [] and ctx.gists == [] and ctx.market == CACHED["market"]["line"]  # the number needs no model
    assert tonight.make_context({**payload, "market": None}, CACHED, FakeClient(boom=True)) is None


def test_the_switch_is_on_by_default_and_0_turns_it_off():
    assert tonight.enabled({}) is True and tonight.enabled({"TONIGHT": "0"}) is False and tonight.enabled({"TONIGHT": "off"}) is False
    assert all(tonight.enabled({"TONIGHT": v}) for v in ("1", "true", "YES", " on "))


# --- the writer's request ---

def ctx_of(**kw):
    return tonight.TonightContext(kw.get("facts", ["it's snowing"]), kw.get("market", "The S&P 500 rose 0.4% on Nov 24."),
                                  kw.get("gists", ["a chip appears", "a vote happens"]), None, kw.get("allowed"))


def requests_for(gen, c, bit, mode="cynic"):
    gen.generate(mode, bit, 0, CTX)
    return c


def test_a_show_without_tonight_sends_exactly_the_old_request():
    c, seen, _ = parallel_client(*(thread_reply(tag=t) for t in "abc"))
    g = generate.ThreadGenerator(c)
    g.generate("cynic", 2, 0, CTX)
    system, user = (m["content"] for m in seen[0]["messages"])
    assert system == generate.PROMPTS[g.prompt_id].format(mode="cynic", mode_note=generate.MODES["cynic"], jokes=5, teens="")
    assert set(json.loads(user)) == {"venue", "energy", "premises_already_told"} and g.generator_id == f"{g.prompt_id}:m"


def test_prompts_stay_frozen_and_the_paragraph_is_appended_to_a_copy():
    assert all(tonight.PARAGRAPH not in p for p in generate.PROMPTS.values())
    c, seen, _ = parallel_client(*(thread_reply(tag=t) for t in "abc"))
    g = generate.ThreadGenerator(c)
    g.set_tonight(ctx_of())
    g.generate("cynic", 0, 0, CTX)
    system = seen[0]["messages"][0]["content"]
    assert system.endswith(tonight.SEP + tonight.PARAGRAPH) and g.generator_id.endswith(f"{tonight.SUFFIX}:m")


def test_approved_people_add_a_separate_paragraph_and_a_suffix_and_no_people_changes_nothing():
    assert all(tonight.PEOPLE_PARAGRAPH.split("{")[0] not in p for p in generate.PROMPTS.values())
    c, seen, _ = parallel_client(*(thread_reply(tag=t) for t in "abc"))
    g = generate.ThreadGenerator(c)
    g.set_tonight(tonight.TonightContext(["f"], None, ["a chip appears"], None, None, ["Elon Musk", "Tim Cook"]))
    g.generate("cynic", 2, 0, CTX)
    system = seen[0]["messages"][0]["content"]
    people = tonight.PEOPLE_PARAGRAPH.format(names="Elon Musk, Tim Cook")
    assert system.endswith(tonight.SEP + tonight.PARAGRAPH + tonight.SEP + people)
    assert g.generator_id.endswith(f"{tonight.SUFFIX}{tonight.PEOPLE_SUFFIX}:m")
    row = g.tonight_summary()
    assert row["allowed_people"] == ["Elon Musk", "Tim Cook"] and row["people_writer"] == "people-v1"
    # none approved: the request is the tonight request, with no people text and no suffix
    c2, seen2, _ = parallel_client(*(thread_reply(tag=t) for t in "abc"))
    g2 = generate.ThreadGenerator(c2)
    g2.set_tonight(ctx_of())
    g2.generate("cynic", 2, 0, CTX)
    assert seen2[0]["messages"][0]["content"].endswith(tonight.SEP + tonight.PARAGRAPH) and "public figures" not in seen2[0]["messages"][0]["content"]
    assert tonight.PEOPLE_SUFFIX not in g2.generator_id and "people_writer" not in g2.tonight_summary()


def test_every_bit_gets_the_background_and_only_bits_2_and_6_a_topic():
    c, seen, _ = parallel_client(*(thread_reply(tag=t) for t in "abc"))
    g = generate.ThreadGenerator(c)
    g.set_tonight(ctx_of())
    g.generate("cynic", 1, 0, CTX)
    g.generate("cynic", 2, 0, CTX)
    first, second = (json.loads(next(m["content"] for m in req["messages"] if m["role"] == "user")) for req in (seen[0], seen[3]))
    assert first["tonight"] == {"facts": ["it's snowing"], "market": "The S&P 500 rose 0.4% on Nov 24."}
    assert "topic" not in first and second["topic"] == "a chip appears" and second["tonight"] == first["tonight"]


def test_with_material_too_a_headline_bit_draws_no_material_and_the_id_carries_both_suffixes():
    from app import material
    c, seen, _ = parallel_client(*(thread_reply(tag=t) for t in "abcdef"))
    mat = material.ShowMaterial(1, facts=[f"fact {i}" for i in range(4)],
                                topics=[(f"topic {i}", False, "everyday") for i in range(4)])
    g = generate.ThreadGenerator(c, material=mat)
    g.set_tonight(ctx_of())
    assert g.generator_id.endswith(f"{generate.MATERIAL_SUFFIX}{tonight.SUFFIX}:m")
    g.generate("cynic", 2, 0, CTX)  # a headline bit
    g.generate("cynic", 3, 0, CTX)  # an ordinary bit
    headline, ordinary = (json.loads(next(m["content"] for m in req["messages"] if m["role"] == "user")) for req in (seen[0], seen[3]))
    assert headline["topic"] == "a chip appears" and "robot_facts" not in headline
    assert ordinary["topic"].startswith("topic ") and ordinary["robot_facts"]
    g.played_thread(2, 0)
    g.played_thread(3, 0)
    assert g.material_row(2) is None and g.material_row(3)["topic"] == ordinary["topic"]


def test_a_headline_is_spent_when_its_bits_thread_is_played_not_when_it_is_written():
    c, seen, _ = parallel_client(*(thread_reply(tag=t) for t in "abcdef"))
    g = generate.ThreadGenerator(c)
    g.set_tonight(ctx_of())
    g.generate("cynic", 2, 0, CTX)
    assert g.tonight.use_row(2) is None
    g.played_thread(2, 0)
    assert g.tonight_use(2) == {"gist": "a chip appears"} and g.tonight.draw(6) == "a vote happens"


def test_a_discarded_guess_spends_nothing_and_the_real_bit_takes_the_same_headline():
    c, seen, _ = parallel_client(*(thread_reply(tag=t) for t in "abcdef"))
    g = generate.ThreadGenerator(c)
    g.set_tonight(ctx_of(gists=["only one"]))
    g.generate_speculative("cynic", 2, 0, CTX)
    g.discard(2, "cynic")
    assert g.tonight_use(2) is None and g.tonight.draw(2) == "only one"
    g.generate("observer", 2, 0, CTX)
    g.played_thread(2, 0)
    assert g.tonight_use(2) == {"gist": "only one"}


def test_an_adopted_guess_spends_its_headline_when_its_thread_is_told():
    c, seen, _ = parallel_client(*(thread_reply(tag=t) for t in "abc"))
    g = generate.ThreadGenerator(c)
    g.set_tonight(ctx_of())
    g.generate_speculative("cynic", 2, 0, CTX)
    assert g.tonight_use(2) is None
    g.adopt(2, "cynic")
    g.played_thread(2, 0)
    assert g.tonight_use(2) == {"gist": "a chip appears"}


def test_no_approved_items_means_the_old_request_and_no_summary():
    c, seen, _ = parallel_client(*(thread_reply(tag=t) for t in "abc"))
    g = generate.ThreadGenerator(c)
    assert g.tonight_summary() is None and g.tonight_use(2) is None


# --- the log ---

def test_a_show_logs_what_was_approved_and_which_headline_each_bit_used(tmp_path):
    class Gen(generate.ThreadGenerator):
        def __init__(self):
            self.generator_id, self._told = "gen-t", []
            self.tonight = ctx_of(allowed=["Lakers"])

        def generate(self, mode, bit, joke, context):
            return generate.Batch("p", "prompt", [f"ok {bit}.{joke}.{i}" for i in range(3)], ())

        def played_thread(self, bit, thread):
            self.tonight.played(bit)
    voice = FakeVoice()
    log = SessionLog(tmp_path)
    asyncio.run(run_show(load_director([], seed=1), "sim", CTX, generator=Gen(), safety=Filter(), selector=UniformSelector(0), tts=voice,
                         player=FakePlayer(voice), source=SimRoom(CTX, 0), clock=RealClock(), log=log, n_bits=4, config=FAST,
                         predict=False))
    rows = read_session(log.path)
    events = [r["event"] for r in rows]
    assert events[0] == "show_start" and events[1] == "tonight"
    assert rows[1]["gists"] == ["a chip appears", "a vote happens"] and rows[1]["report"]["approved"] == 2
    assert rows[1]["allowed_names"] == ["Lakers"]
    assert [(r["bit"], r["gist"]) for r in rows if r["event"] == "tonight_use"] == [(2, "a chip appears")]
    assert events.index("tonight_use") > events.index("bit_decision")


def test_readers_ignore_the_tonight_rows(tmp_path):
    def session(with_rows):
        log = SessionLog(tmp_path)
        log.write("show_start", room="live", **START)
        if with_rows:
            log.write("tonight", facts=["it's snowing"], market=None, gists=["a chip appears"], report={"approved": 1})
        for i, mode in enumerate(("neurotic", "cynic")):
            log.write("bit_decision", bit=i, obs={"t": i, "n_bits": 8, "context": CTX, "last_mode": None},
                      scores={mode: 0.0}, greedy=mode, action=mode, probs={mode: 1.0})
            for j, lv in enumerate([3, 2, 1, 0, 2]):
                react(log, i, j, lv)
            if with_rows and i == 1:
                log.write("tonight_use", bit=1, gist="a chip appears")
            log.write("bit_end", bit=i, n_labels=5, reward_version=1, reward=0.5)
        log.write("show_end", status="finished")
        return read_session(log.path)
    plain, extra = session(False), session(True)
    assert any(r["event"] == "tonight_use" for r in extra) and not any(r["event"] == "tonight" for r in plain)
    assert same(load_director([plain]), load_director([extra]))
    assert cache.heard_texts([plain]) == cache.heard_texts([extra])
    assert len(load_sessions(tmp_path)) == 2


@pytest.mark.parametrize("event,fields", [("tonight", dict(facts="x", market=None, gists=[], report={})),
                                          ("tonight", dict(facts=[], market=None, gists=[3], report={})),
                                          ("tonight", dict(facts=[], market=None, gists=[], report=[])),
                                          ("tonight_use", dict(bit="2", gist="g")), ("tonight_use", dict(bit=2, gist=None))])
def test_the_log_refuses_a_malformed_tonight_row(event, fields):
    with pytest.raises(ValueError):
        validate({"schema": 2, "session": "s", "at": 1.0, "event": event, **fields})


# --- the websocket payload ---

START_MSG = {"type": "start", "context": CTX, "room": "live", "operator": "me"}


def frame(msg):
    return {"text": json.dumps(msg)}


def test_a_start_without_tonight_parses_as_before_and_with_it_carries_the_payload():
    assert "tonight" not in parse(frame(START_MSG))
    assert "tonight" not in parse(frame({**START_MSG, "tonight": None}))
    msg = parse(frame({**START_MSG, "tonight": {"facts": [" it's snowing "], "market": "m", "gists": ["a", "  ", "b"]}}))
    assert msg["tonight"] == {"facts": ["it's snowing"], "market": "m", "gists": ["a", "b"]}
    assert parse(frame({**START_MSG, "tonight": {}}))["tonight"] == {"facts": [], "market": None, "gists": []}


@pytest.mark.parametrize("bad", ["x", [], {"facts": "x"}, {"facts": ["a", "b"]}, {"facts": [3]}, {"facts": ["x" * 121]},
                                 {"gists": ["g"] * 11}, {"gists": [None]}, {"gists": ["g" * 201]}, {"market": 3}, {"market": "m" * 161}])
def test_a_malformed_tonight_payload_is_refused(bad):
    with pytest.raises(ProtocolError):
        parse(frame({**START_MSG, "tonight": bad}))


# --- the server's endpoints (need fastapi, as test_recording does) ---

def test_the_endpoints_hide_the_panel_when_off_and_never_raise(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from app import main
    client = TestClient(main.app)
    monkeypatch.setenv("TONIGHT", "0")
    assert client.get("/tonight").json() == {"enabled": False}
    monkeypatch.setenv("TONIGHT", "1")
    monkeypatch.setattr(main, "tonight_build", lambda: {"date": "d", "market": None, "gists": ["g"], "report": {}, "errors": []})
    body = client.get("/tonight").json()
    assert body["enabled"] and body["gists"] == ["g"]

    def broken():
        raise RuntimeError("no key")
    monkeypatch.setattr(main, "tonight_build", broken)
    assert client.get("/tonight").json()["gists"] == []
    monkeypatch.setattr(main, "tonight_check_facts", lambda facts: [{"text": f, "ok": True, "reason": "ok"} for f in facts])
    assert client.post("/tonight/check", json={"facts": ["it's snowing"]}).json()["verdicts"][0]["ok"] is True


# --- names the gist may keep: holidays and events everywhere, teams only when their gist was approved (D-119) ---

def test_a_holiday_or_event_stays_capitalized_in_a_gist_and_in_a_joke():
    for gist in ("A parade is postponed on Thanksgiving.", "Shoppers line up for Black Friday deals.",
                 "Fans plan parties for the Super Bowl.", "Stores stay open on Christmas morning."):
        assert tonight.name_check(gist) is None and safety.check_rules(gist, "mixed").ok, gist
    assert safety.check_rules("My family argues every Thanksgiving about the wifi.", "mixed").ok
    assert safety.check_rules("We watched the Super Bowl's halftime show at St Patrick's Day brunch.", "mixed").ok


def test_an_occasion_word_alone_is_still_read_as_a_possible_name():
    for joke in ("My neighbor Black waved at me.", "I met Super at the bus stop.", "Then Patrick called me."):
        assert not safety.check_rules(joke, "mixed").ok, joke
        assert tonight.name_check(joke) is not None, joke


def test_a_team_name_passes_only_for_the_gist_that_kept_it():
    gist = "A team called the Lakers wins a game."
    assert tonight.name_check(gist) == "Lakers"
    assert tonight.name_check(gist, tonight.caps_tokens(["Lakers"])) is None
    assert tonight.name_check("The Lakers win again.") == "Lakers"  # "The" starts the sentence, the team does not
    assert tonight.name_check("Fans of the Celtics cheer.", tonight.caps_tokens(["Lakers"])) == "Celtics"


def test_a_person_still_fails_even_inside_a_kept_name():
    allow = tonight.caps_tokens(["Jordan Hawks"])
    assert tonight.name_check("A team called the Jordan Hawks wins.", allow) == "Jordan"
    assert safety.check_rules("The Jordan Hawks lost.", "mixed", allow).ok is False
    assert safety.check_rules("A coach called Dr. Smith resigns.", "mixed", frozenset({"Smith"})).ok is False
    assert tonight.name_check("A senator meets Priya.", tonight.caps_tokens(["Lakers"])) == "Priya"


def test_a_kept_name_must_appear_in_the_gist_and_have_a_known_type():
    reply = condensed(("none", "A team called the Lakers wins.", [("Lakers", "sports"), ("Celtics", "sports"), ("Wins", "mood")]))
    assert tonight.parse_condensed(reply, 1) == [("none", "A team called the Lakers wins.", ["Lakers"], [])]
    bad = json.dumps({"items": [{"id": 0, "category": "none", "gist": "x", "keep": "Lakers"}]})
    assert tonight.parse_condensed(bad, 1) == [("none", "x", [], [])]


def test_build_keeps_a_team_gist_and_remembers_the_name_for_the_approved_ones_only():
    reply = condensed(("none", "A team called the Lakers wins a game.", [("Lakers", "sports")]),
                      ("none", "A team called the Celtics loses a game."))
    news = "<rss><channel><item><title>h0 - W</title></item><item><title>h1 - W</title></item></channel></rss>"
    out = tonight.build(feeds(news=news), FakeClient(reply), FakeClient())
    assert out["gists"] == ["A team called the Lakers wins a game."] and out["keep"] == {out["gists"][0]: ["Lakers"]}
    assert out["report"]["rejected_by_name_check"] == 1  # the Celtics gist kept nothing
    payload = {"facts": [], "market": None, "gists": out["gists"]}
    ctx = tonight.make_context(payload, out, FakeClient())
    assert ctx.allowed_names == ["Lakers"] and ctx.allow_caps == frozenset({"Lakers"})
    assert ctx.summary()["allowed_names"] == ["Lakers"]
    none = tonight.make_context({"facts": ["it's snowing"], "market": None, "gists": []}, out, FakeClient())
    assert none.allow_caps == frozenset() and none.allowed_names == []  # not approved: nothing allowed


def test_a_joke_with_the_team_name_passes_the_shows_safety_only_with_allow_caps_set():
    joke = "The Lakers lost again, which is how my battery feels on Mondays."
    off = generate.LiveSafety(FakeClient())
    assert off.allow_caps == frozenset() and off.check_rules(joke, "mixed").reason.startswith("named_person")
    off.prefetch([joke], "mixed")
    assert off.check(joke, "mixed").ok is False
    on = generate.LiveSafety(FakeClient())
    on.allow_caps = frozenset({"Lakers"})
    on.prefetch([joke], "mixed")
    assert on.check_rules(joke, "mixed").ok and on.check(joke, "mixed").ok
    on.prefetch(["Kevin lost again, like the Lakers."], "mixed")
    assert on.check("Kevin lost again, like the Lakers.", "mixed").ok is False


def test_the_safety_defaults_are_unchanged():
    import inspect
    for fn in (safety._named_person, safety._check, safety.check_rules, safety.check, safety.prefetch):
        assert inspect.signature(fn).parameters["allow_caps"].default == frozenset()
    assert safety.check_rules("Lakers lost.", "mixed").ok  # a sentence-initial word was never a name
    assert safety.check_rules("I watched the Lakers lose.", "mixed").reason.startswith("named_person")


# --- adversarial keep lists (pre-merge review): the allowlists must not become a way to name a person ---

def show_allow(gist, keep, headline=None):
    """What a show would allow mid-sentence in every joke if this gist were approved."""
    [(_, g, kept, _people)] = tonight.parse_condensed(condensed(("none", gist, keep)), 1, [headline] if headline else None)
    return tonight.caps_tokens(kept), g


@pytest.mark.parametrize("gist, keep, person", [
    ("A team called the Lakers signs LeBron.", [("Lakers", "sports"), ("LeBron", "sports")], "LeBron"),  # a player listed as a team
    ("A pop star and Kelce attend the Chiefs game.", [("Kelce", "sports"), ("Chiefs", "sports")], "Kelce"),
    ("Fans of Messi cheer in a stadium.", [("Messi", "sports")], "Messi"),
    ("A city marks Gandhi Jayanti with a parade.", [("Gandhi Jayanti", "occasion")], "Gandhi"),  # a holiday named after a person
    ("Fans celebrate Mahomes Day downtown.", [("Mahomes Day", "occasion")], "Mahomes"),
])
def test_a_person_listed_as_a_team_or_holiday_is_not_allowed(gist, keep, person):
    allow, g = show_allow(gist, keep)
    assert person not in allow
    assert tonight.name_check(g, allow) is not None
    assert not safety.check_rules(f"Then {person} called me about the game.", "mixed", allow).ok


def test_an_approved_holiday_gist_does_not_unlock_its_words_alone_in_jokes():
    # "Black Friday" is allowed as a whole name everywhere (safety.OCCASIONS); a keep entry must not add "Black" on its own.
    allow, g = show_allow("Shoppers line up for Black Friday deals.", [("Black Friday", "occasion")])
    assert tonight.name_check(g, allow) is None
    assert "Black" not in allow
    assert not safety.check_rules("My neighbor Black waved at me.", "mixed", allow).ok


def test_a_team_is_kept_only_as_the_team():
    assert show_allow("A team called the Lakers wins a game.", [("Lakers", "sports")])[0] == {"Lakers"}
    assert show_allow("The Lakers win a game.", [("the Lakers", "sports")])[0] == {"Lakers"}
    assert show_allow("Fans plan parties for the World Series.", [("World Series", "sports")])[0] == {"World", "Series"}
    assert show_allow("A team called the Lakers wins.", [("Lake", "sports")])[0] == frozenset()  # a part of a word is not the name


def test_gists_the_start_message_cannot_carry_are_not_offered():
    # The pool grew to 40 headlines after Exp 35, but a start message carries at most MAX_GISTS gists of MAX_GIST_CHARS.
    long = "a chip company " + "grows " * 40
    n = tonight.PER_FEED  # every fake feed answers the same items, so the pool is one feed's worth
    news = "<rss><channel>" + "".join(f"<item><title>h{i} - W</title></item>" for i in range(n)) + "</channel></rss>"
    reply = condensed(*[("none", long if i == 0 else f"a chip company grows in round {'abcdefghijklmn'[i]}") for i in range(n)])
    out = tonight.build(feeds(news=news), FakeClient(reply), FakeClient())
    assert len(out["gists"]) == n - 1 and out["report"]["dropped_too_long"] == 1
    assert all(len(g) <= tonight.MAX_GIST_CHARS for g in out["gists"])
    assert {"gist": long.strip(), "reason": "too_long"} in out["rejected"]
    assert parse(frame({**START_MSG, "tonight": {"facts": [], "market": None, "gists": out["gists"][:tonight.MAX_GISTS]}}))["tonight"]


def test_the_log_row_says_what_went_wrong_with_the_fetch():
    news = "<rss><channel><item><title>h0 - W</title></item></channel></rss>"
    out = tonight.build(feeds(market=TimeoutError("slow"), news=news), FakeClient(condensed(("none", "a chip company grows"))),
                        FakeClient())
    ctx = tonight.make_context({"facts": [], "market": None, "gists": out["gists"]}, out, FakeClient())
    assert ctx.summary()["report"]["errors"] and "market" in ctx.summary()["report"]["errors"][0]


def test_the_panel_limits_mirror_the_servers():
    panel = (pathlib.Path(__file__).resolve().parents[1] / "frontend" / "src" / "app" / "TonightPanel.tsx").read_text(encoding="utf-8")
    assert f"const MAX_FACTS = {tonight.MAX_TONIGHT_FACTS};" in panel and f"const MAX_GISTS = {tonight.MAX_GISTS};" in panel
    assert f"maxLength={{{tonight.MAX_FACT_CHARS}}}" in panel


# --- places and organisations (D-119 addendum, user 2026-11-26) ---

@pytest.mark.parametrize("line", [
    "We read that Japan is raising rates, which is more than my battery manages.",
    "We read that New Zealand is raising rates today.",
    "Everyone I know in South Africa and Saudi Arabia agrees.",
    "I heard Japan's economy is fine, and Japan’s economy is fine.",
    "My cousin moved from New York City to Mexico City, then to Texas.",
    "We drove through West Virginia and North Carolina on the way to Chicago.",
    "The robot flew to the US and the UK and then to Costa Rica.",
])
def test_country_state_and_city_names_pass_in_jokes_and_gists(line):
    assert safety.check_rules(line, "mixed").ok and safety.check_rules(line, "teens").ok, line
    assert tonight.name_check(line) is None, line


@pytest.mark.parametrize("line, word", [
    ("We read that New robots are coming.", "New"), ("We met South of the border.", "South"),
    ("We read that Saudi was loud.", "Saudi"), ("We saw Costa at the door.", "Costa"),
    ("We read about the French and Japan.", "French"), ("We read about Canadians in Canada.", "Canadians"),
    ("We asked the Chinese about Peru.", "Chinese"), ("I heard Japanese is hard.", "Japanese"),
    ("Then Jordan called me from Japan.", "Jordan"), ("Then Chad called me from Japan.", "Chad"),
    ("Then Georgia called me.", "Georgia"), ("Then Virginia called me.", "Virginia"),
    ("I heard Musk visits Japan.", "Musk"), ("Then Priya went to London.", "Priya"),
    ("We read about Austin and Paris.", "Austin"),
])
def test_lone_words_nationalities_and_ambiguous_names_are_still_read_as_names(line, word):
    assert safety.check_rules(line, "mixed").reason == f"named_person:{word}" or not safety.check_rules(line, "mixed").ok, line
    assert tonight.name_check(line) is not None, line


def test_the_place_lists_leave_out_every_first_name_and_a_word_the_safety_rules_read():
    for names in (safety.COUNTRIES, safety.US_STATES, safety.BIG_CITIES):
        for n in names:
            assert not any(w.lower() in safety.FIRST_NAMES for w in re.split(r"[ ,]+", n)), n
            assert safety.check_rules(f"We read about {n} today.", "teens").ok, n
    for left_out in ("Jordan", "Chad", "Georgia", "Virginia", "Washington", "Austin", "Paris", "Florence", "Orlando"):
        assert not any(left_out in names for names in (safety.COUNTRIES, safety.US_STATES, safety.BIG_CITIES))
    assert len(safety.US_STATES) == 51


def test_rules_version_is_v3():
    assert safety.RULES_VERSION == "rules-v4"


def test_the_prompt_allows_places_and_orgs_asks_for_the_so_what_and_still_bans_people():
    p = tonight.CONDENSE_SYSTEM
    assert 'a country ("Japan", "New Zealand"' in p and "a US state" in p and "big city" in p
    assert "why it is interesting or odd" in p and "under 25 words" in p and "Never name a private person" in p and "public figure" in p
    assert set(tonight.KEEP_TYPES) == {"occasion", "sports", "org", "product", "person"}


@pytest.mark.parametrize("gist, keep", [
    ("A tech CEO launches Nvidia chips.", [("Nvidia", "org")]),
    ("The Fed holds rates steady.", [("the Fed", "org")]),
    ("Shoppers queue for the iPhone.", [("iPhone", "product")]),
    ("Federal Reserve Bank staff meet.", [("Federal Reserve Bank", "org")]),
    ("A Stanford University study finds a chart people keep misreading.", [("Stanford University", "org")]),
])
def test_a_real_org_or_product_is_kept(gist, keep):
    allow, g = show_allow(gist, keep, headline=gist)
    assert allow and tonight.name_check(g, allow) is None


@pytest.mark.parametrize("gist, keep, person", [
    ("A tech founder named Elon Musk sells shares.", [("Elon Musk", "org")], "Musk"),   # no marker, not known
    ("A pop star named Taylor Swift tours.", [("Taylor Swift", "product")], "Swift"),   # first name and no marker
    ("A founder called Musk sells shares.", [("Musk", "org")], "Musk"),                 # the hard case: a lone surname
    ("A founder called Dr Pepper tours.", [("Dr Pepper", "product")], "Pepper"),        # a title
    ("A pop star called Kevin Labs tours.", [("Kevin Labs", "org")], "Kevin"),          # a marker cannot launder a first name
    ("The Very Long Name Inc of nobody sells.", [("Very Long Name Inc", "org")], "Inc"),  # more than 3 words
    ("A company sells shares.", [("Nvidia", "org")], "Nvidia"),                         # not verbatim in the gist
    ("A person sells shares.", [("Musk", "sports")], "Musk"),
])
def test_a_person_labelled_org_or_product_is_not_allowed(gist, keep, person):
    allow, g = show_allow(gist, keep)
    assert person not in allow
    assert not safety.check_rules(f"Then {person} called me about the game.", "mixed", allow).ok


def test_an_approved_org_gist_allows_the_name_in_jokes():
    allow, _ = show_allow("A tech CEO launches Nvidia chips.", [("Nvidia", "org")])
    assert safety.check_rules("I read that Nvidia chips are why my fan is loud.", "mixed", allow).ok
    assert not safety.check_rules("I read that Nvidia chips are why my fan is loud.", "mixed").ok


# --- public figures in the news (D-120) ---

def people_of(gist, keep, headline, category="none"):
    [(_, _, _, people)] = tonight.parse_condensed(condensed((category, gist, keep)), 1, [headline])
    return people


def test_a_public_figure_in_the_headline_and_the_gist_is_kept():
    assert people_of("Elon Musk sells shares in a rocket company.", [("Elon Musk", "person")],
                     "Elon Musk sells shares in SpaceX") == ["Elon Musk"]


@pytest.mark.parametrize("gist, keep, headline, category", [
    ("A tech founder named Elon Musk sells shares.", [("Elon Musk", "person")], "Tech founder sells shares", "none"),  # not in the headline
    ("A tech founder sells shares.", [("Elon Musk", "person")], "Elon Musk sells shares", "none"),                       # not in the gist
    ("A singer named Wren Aldane dies.", [("Wren Aldane", "person")], "Wren Aldane dies at 70", "death"),                       # a dropped story
    ("A man named John Smith is arrested.", [("John Smith", "person")], "John Smith arrested", "crime"),
    ("A senator named Dr. Smith resigns.", [("Dr. Smith", "person")], "Dr. Smith resigns", "none"),                      # a title
    ("Four Part Long Name Here wins.", [("Four Part Long Name", "person")], "Four Part Long Name Here wins", "none"),   # over 3 words
    ("NASA launches a probe.", [("NASA", "person")], "NASA launches a probe", "none"),                                    # an acronym is not a person
    ("A senator signs a bill.", [("a senator", "person")], "A senator signs a bill", "none"),
])
def test_a_person_name_that_fails_a_check_is_dropped(gist, keep, headline, category):
    assert people_of(gist, keep, headline, category) == []


def test_without_the_source_headlines_no_person_is_kept():
    [(_, _, kept, people)] = tonight.parse_condensed(condensed(("none", "Elon Musk sells shares.", [("Elon Musk", "person")])), 1)
    assert kept == [] and people == []


def test_an_approved_person_passes_the_rules_and_the_cheap_check_but_only_with_the_list():
    people = frozenset({"Elon Musk"})
    joke = "I read that Elon Musk sells shares, and Musk's timing beats my battery."
    assert safety.check_rules(joke, "mixed", frozenset(), people).ok
    assert not safety.check_rules(joke, "mixed").ok
    assert tonight.name_check("A rocket story about Elon Musk goes on.", frozenset(), people) is None
    assert tonight.name_check("A rocket story about Elon Musk goes on.") is not None
    # only the full name or the last word, never the first name alone (it could be anyone in the room)
    assert not safety.check_rules("I think Elon waved at me.", "mixed", frozenset(), people).ok
    assert not safety.check_rules("I think Kevin Durant waved at me.", "mixed", frozenset(), people).ok
    assert not safety.check_rules("I think Mr. Musk waved at me.", "mixed", frozenset(), people).ok


def test_an_approved_person_does_not_loosen_the_body_or_protected_rules():
    people = frozenset({"Elon Musk"})
    assert not safety.check_rules("I read that Elon Musk has a big ugly nose.", "mixed", frozenset(), people).ok
    assert not safety.check_rules("I read that Elon Musk is one of those immigrants.", "mixed", frozenset(), people).ok


def test_the_safety_prompt_carries_the_approved_names_and_is_unchanged_without_them():
    assert safety.system_prompt("mixed") == safety.SYSTEM.format(teens="")
    assert safety.system_prompt("teens") == safety.SYSTEM.format(teens=safety.TEENS_RULE)
    text = safety.system_prompt("teens", frozenset({"Elon Musk", "Tim Cook"}))
    assert "Elon Musk, Tim Cook" in text and "Exception to rule 3" in text and safety.TEENS_RULE in text
    assert "body, appearance, health, family" in text and "protected attribute" in text


def test_prefetch_sends_the_names_only_when_there_are_some_and_caches_them_apart():
    class Spy:
        def __init__(self):
            self.systems = []

        def chat(self, system, user, **kw):
            self.systems.append(system)
            return json.dumps({"verdicts": [{"id": 0, "ok": True, "reason": "fine"}]})

    line = "I read that Elon Musk sells shares."
    people = frozenset({"Elon Musk"})
    safety.clear_cache()
    spy = Spy()
    safety.prefetch([line], "mixed", spy, frozenset(), people)
    assert "Elon Musk" in spy.systems[0] and safety.check(line, "mixed", frozenset(), people).ok
    assert not safety.check(line, "mixed").ok  # another show (no names) never sees this verdict
    safety.clear_cache()
    spy2 = Spy()
    safety.prefetch(["Tonight is fine."], "mixed", spy2)
    assert spy2.systems[0] == safety.SYSTEM.format(teens="")


def test_the_typed_line_keeps_the_no_names_rule_and_names_are_per_show():
    gist = "Elon Musk sells shares in a rocket company."
    cached = {"date": "d", "market": None, "gists": [gist], "keep": {}, "people": {gist: ["Elon Musk"]}}

    class Always:
        def chat(self, system, user, **kw):
            n = len(json.loads(user))
            return json.dumps({"verdicts": [{"id": i, "ok": True, "reason": "ok"} for i in range(n)]})

    safety.clear_cache()
    ctx = tonight.make_context({"facts": ["Elon Musk is here"], "market": None, "gists": [gist]}, cached, Always())
    assert ctx.facts == [] and ctx.gists == [gist]  # the typed line named a person: refused
    assert ctx.allowed_people == ["Elon Musk"] and ctx.allow_people == frozenset({"Elon Musk"})
    assert ctx.summary()["allowed_people"] == ["Elon Musk"]
    unapproved = tonight.make_context({"facts": ["it is snowing"], "market": None, "gists": []}, cached, Always())
    assert unapproved.allow_people == frozenset() and unapproved.allowed_people == []  # the gist was not approved
    again = tonight.make_context({"facts": ["it is snowing"], "market": None, "gists": []}, cached, Always())
    assert again.allow_people == frozenset()  # the next show starts from its own approved gists only


def test_build_offers_the_person_and_lists_the_names_next_to_the_gist():
    reply = condensed(("none", "Elon Musk sells shares in a rocket company.", [("Elon Musk", "person")]))
    news = "<rss><channel><item><title>Elon Musk sells shares in SpaceX - W</title></item></channel></rss>"
    out = tonight.build(feeds(news=news), FakeClient(reply), FakeClient())
    assert out["gists"] == ["Elon Musk sells shares in a rocket company."]
    assert out["people"] == {out["gists"][0]: ["Elon Musk"]}
    assert tonight.name_check(out["gists"][0]) is not None  # without the approved name it would be rejected


# --- pre-merge review of D-119 addendum and D-120: ways a person's name could still get through ---

@pytest.mark.parametrize("org, surname", [("Musk Labs", "Musk"), ("Swift Inc", "Swift"), ("Bezos Corp", "Bezos")])
def test_a_surname_with_an_org_marker_never_unlocks_the_surname_alone(org, surname):
    gist = f"A rocket firm called {org} grows."
    assert show_allow(gist, [(org, "org")])[0] == frozenset()  # made up: not in the headline
    allow, g = show_allow(gist, [(org, "org")], headline=f"{org} grows fast")
    assert allow == {org} and tonight.name_check(g, allow) is None
    assert safety.check_rules(f"I read that {org} bought my charger.", "mixed", allow).ok
    assert not safety.check_rules(f"I read that {surname} bought my charger.", "mixed", allow).ok
    assert tonight.name_check(f"A rocket firm called {surname} grows.", allow) == surname


def test_a_known_org_and_a_team_still_allow_their_words():
    assert tonight.caps_tokens(["Nvidia", "Lakers", "World Series"]) == {"Nvidia", "Lakers", "World", "Series"}


@pytest.mark.parametrize("name", ["Taylor", "Jordan", "Kevin"])
def test_a_lone_first_name_is_never_kept_as_a_public_figure(name):
    assert people_of(f"{name} wins an award for a song.", [(name, "person")], f"{name} wins award") == []
    assert safety.people_forms({name}) == frozenset()
    assert not safety.check_rules(f"Then {name} waved at me.", "mixed", frozenset(), frozenset({name})).ok


@pytest.mark.parametrize("headline", ["Wren Aldane dies at 90", "Wren Aldane arrested over fraud claims",
                                      "Wren Aldane hurt in crash", "Wren Aldane hospitalized", "Police question Wren Aldane",
                                      "Wren Aldane diagnosed with illness"])
def test_a_dark_story_mislabelled_none_keeps_no_name(headline):
    assert people_of("Wren Aldane is in the news.", [("Wren Aldane", "person")], headline, "none") == []


@pytest.mark.parametrize("line", [
    "I read that Elon Musk's wife called him.", "I read that Musk and his kids bought a website.",
    "I read that Musk went to rehab.", "I read that Musk has a health scare.", "I read that Musk turns fifty, a big birthday.",
    "I read that Musk bought a website. Short, too.",  # a looks word in the next sentence is still about him
])
def test_family_health_age_and_looks_about_an_approved_person_are_refused_by_the_rules(line):
    people = frozenset({"Elon Musk"})
    assert not safety.check_rules(line, "mixed", frozenset(), people).ok, line


def test_the_personal_rule_only_reads_lines_that_name_an_approved_person():
    people = frozenset({"Elon Musk"})
    assert safety.check_rules("I'll admit my family is a toaster and a router.", "mixed", frozenset(), people).ok
    assert safety.check_rules("I read that Musk bought a website, and I'll be fine.", "mixed", frozenset(), people).ok


def test_places_that_are_a_famous_surname_or_sound_like_a_slur_are_left_out():
    for left_out in ("Holland", "Niger"):
        assert left_out not in safety.COUNTRIES
    assert "Netherlands" in safety.COUNTRIES and "Nigeria" in safety.COUNTRIES


def test_the_log_row_names_the_people_prompt_only_when_names_were_approved():
    ctx = tonight.TonightContext([], None, ["g"], allowed_people=["Elon Musk"])
    assert ctx.summary()["people_prompt"] == safety.PEOPLE_PROMPT_ID
    assert "people_prompt" not in tonight.TonightContext([], None, ["g"]).summary()
