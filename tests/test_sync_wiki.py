"""docs/wiki -> GitHub wiki link rewriting."""
from scripts.sync_wiki import BLOB, convert, rewrite_link

NAMES = {"README.md": "Home", "glossary.md": "Glossary", "contextual-bandits.md": "Contextual-bandits"}


def test_wiki_to_wiki_links_become_page_names_and_keep_anchors():
    assert rewrite_link("glossary.md", NAMES) == "Glossary"
    assert rewrite_link("glossary.md#learning-how-the-director-chooses", NAMES) == \
        "Glossary#learning-how-the-director-chooses"
    assert rewrite_link("README.md", NAMES) == "Home"


def test_repo_links_become_blob_urls():
    assert rewrite_link("../metrics.md", NAMES) == BLOB + "docs/metrics.md"
    assert rewrite_link("../design.md#long-form-set-generation", NAMES) == \
        BLOB + "docs/design.md#long-form-set-generation"
    assert rewrite_link("../../pipeline/wiki/README.md", NAMES) == BLOB + "pipeline/wiki/README.md"


def test_external_and_anchor_links_untouched():
    for t in ("https://arxiv.org/abs/1003.0146", "#reading-order"):
        assert rewrite_link(t, NAMES) == t


def test_convert_drops_the_h1_and_rewrites_inline():
    out = convert("# Glossary\n\nSee [bandits](contextual-bandits.md) and [log](../dev-log.md).\n", NAMES)
    assert not out.startswith("# ")
    assert "[bandits](Contextual-bandits)" in out
    assert f"[log]({BLOB}docs/dev-log.md)" in out


def test_inline_images_point_at_the_raw_file_and_links_to_images_stay_blob():
    out = convert("# Page\n\n![table](../../reports/figures/dpo/b.png)\n\n[log](../../reports/figures/dpo/a.png)\n"
                  "![ext](https://example.com/x.png)\n", NAMES)
    assert f"![table]({BLOB}reports/figures/dpo/b.png?raw=true)" in out
    assert f"[log]({BLOB}reports/figures/dpo/a.png)" in out
    assert "![ext](https://example.com/x.png)" in out
