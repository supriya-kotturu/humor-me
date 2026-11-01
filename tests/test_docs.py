"""Checks on the markdown docs themselves."""
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
LIGHT = '%%{init: {"theme": "default"}}%%'


def tracked_markdown():
    out = subprocess.run(["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True)
    return [ROOT / p for p in out.stdout.split()]


def test_mermaid_diagrams_force_light_theme():
    # GitHub renders Mermaid dark in dark mode, which washes out the pastel component colours.
    missing = []
    for md in tracked_markdown():
        lines = md.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "```mermaid" and (i + 1 >= len(lines) or lines[i + 1].strip() != LIGHT):
                missing.append(f"{md.relative_to(ROOT)}:{i + 1}")
    assert not missing, f"Mermaid blocks without {LIGHT}: {missing}"
