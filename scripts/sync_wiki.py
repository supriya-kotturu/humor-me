"""Publish docs/wiki/ to the repo's GitHub wiki (humor-me.wiki.git).

Usage:
    git clone https://github.com/supriya-kotturu/humor-me.wiki.git <dir>
    python scripts/sync_wiki.py <dir>
    cd <dir> && git add -A && git commit -m "Sync from docs/wiki @ <sha>" && git push

docs/wiki/ stays the source of truth (reviewed in PRs); the GitHub wiki is a generated copy so
it's browsable on github.com. Never edit the wiki on the web: the next sync overwrites it.

Conversions: README.md -> Home; other pages are named from their H1; links between wiki pages
become wiki links; links to other repo files become github.com/blob/main URLs.
"""
import pathlib
import posixpath
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "wiki"
BLOB = "https://github.com/supriya-kotturu/humor-me/blob/main/"
LINK = re.compile(r"\]\(([^)\s]+)\)")
IMAGE = re.compile(r"(!\[[^\]]*\]\()([^)\s]+)\)")
FOOTER = ("Generated from [`docs/wiki/`](" + BLOB + "docs/wiki) by `scripts/sync_wiki.py`. "
          "Edit there through a PR; changes made here are overwritten on the next sync.\n")


def page_name(md: pathlib.Path) -> str:
    if md.name == "README.md":
        return "Home"
    title = md.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
    return re.sub(r"[\s/]+", "-", title)


def rewrite_link(target: str, names: dict[str, str], src_dir: str = "docs/wiki") -> str:
    if target.startswith(("http://", "https://", "mailto:", "#")):
        return target
    path, _, anchor = target.partition("#")
    resolved = posixpath.normpath(posixpath.join(src_dir, path))
    suffix = f"#{anchor}" if anchor else ""
    if posixpath.dirname(resolved) == src_dir and resolved.endswith(".md"):
        return names[posixpath.basename(resolved)] + suffix
    return BLOB + resolved + suffix


def image_src(target: str, names: dict[str, str]) -> str:
    """A blob URL serves an HTML page, so an inline image needs the raw file."""
    url = rewrite_link(target, names)
    return url + "?raw=true" if url.startswith(BLOB) else url


def convert(text: str, names: dict[str, str]) -> str:
    body = text.split("\n", 1)[1].lstrip("\n") if text.startswith("# ") else text  # wiki shows the title
    body = IMAGE.sub(lambda m: f"{m.group(1)}{image_src(m.group(2), names)})", body)
    return LINK.sub(lambda m: f"]({rewrite_link(m.group(1), names)})", body)


def sidebar(names: dict[str, str]) -> str:
    """Pages in the order the index links them (the report's story order)."""
    order = []
    for target in LINK.findall((SRC / "README.md").read_text(encoding="utf-8")):
        base = target.partition("#")[0]
        if base in names and names[base] not in order and base != "README.md":
            order.append(names[base])
    titles = {n: (SRC / f).read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
              for f, n in names.items()}
    return "**humor-me wiki**\n\n- [[Home]]\n" + "".join(f"- [[{titles[n]}|{n}]]\n" for n in order)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = pathlib.Path(sys.argv[1])
    if not (out / ".git").exists():
        sys.exit(f"{out} is not a git checkout of the wiki")
    pages = sorted(SRC.glob("*.md"))
    names = {p.name: page_name(p) for p in pages}
    for p in pages:
        (out / f"{names[p.name]}.md").write_text(convert(p.read_text(encoding="utf-8"), names), encoding="utf-8")
    (out / "_Sidebar.md").write_text(sidebar(names), encoding="utf-8")
    (out / "_Footer.md").write_text(FOOTER, encoding="utf-8")
    generated = {f"{n}.md" for n in names.values()} | {"_Sidebar.md", "_Footer.md"}
    extra = sorted(p.name for p in out.glob("*.md") if p.name not in generated)
    print(f"wrote {len(pages)} pages + sidebar + footer to {out}")
    if extra:
        print("pages in the wiki that aren't in docs/wiki (left alone):", ", ".join(extra))


if __name__ == "__main__":
    main()
