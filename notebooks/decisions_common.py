"""Setup shared by the decisions-*.ipynb notebooks: repo root on the path, cwd at the root, claim checks."""
import json
import os
import re
import sys
from pathlib import Path

from IPython.display import Markdown, display

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

RERUN = False  # True = recompute every experiment (~6 min)


def show(md_text):
    display(Markdown(md_text))


def check(claim, ok):
    show(f"{'✅' if ok else '❌'} **{claim}**")
    return ok


def sections(root=ROOT):
    """Experiment number -> [(section heading, notebook file name)], over every decisions-*.ipynb."""
    found = {}
    for path in sorted((root / "notebooks").glob("decisions-*.ipynb")):
        for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
            m = re.match(r"##\s*(Exp (\d+) [^\n]*)", "".join(cell["source"])) if cell["cell_type"] == "markdown" else None
            if m:
                found.setdefault(int(m.group(2)), []).append((m.group(1), path.name))
    return found
