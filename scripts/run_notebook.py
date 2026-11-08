"""Restart + run all for the decision notebooks, in place, with clean metadata. Run before committing them.

Usage:
    .venv/Scripts/python scripts/run_notebook.py                      # every notebooks/decisions*.ipynb, then decision-plot
    .venv/Scripts/python scripts/run_notebook.py notebooks/decisions-data.ipynb [--timeout 1200]

Runs with the interpreter that launches this script (the "python3" kernel resolves to it), so use
the project venv. Outputs are kept (GitHub renders them); timing metadata isn't recorded, so
re-running without changes produces no diff beyond the outputs themselves. Exits 1 if any claim check fails.
"""
import argparse
import pathlib
import sys

import nbformat
from nbclient import NotebookClient

NOTEBOOKS = pathlib.Path(__file__).resolve().parent.parent / "notebooks"


def run(path: pathlib.Path, timeout: int) -> list[str]:
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=timeout, kernel_name="python3", record_timing=False,
                   resources={"metadata": {"path": str(path.parent)}}).execute()
    for cell in nb.cells:
        cell.metadata.pop("execution", None)
    nb.metadata.pop("widgets", None)
    nb.metadata["kernelspec"] = {"name": "humor-me", "display_name": "humor-me (.venv)", "language": "python"}
    nbformat.write(nb, path)
    return [line for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])
            for line in o.get("data", {}).get("text/markdown", "").splitlines() if "❌" in line]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--timeout", type=int, default=1200)
    a = ap.parse_args()
    paths = [pathlib.Path(a.path)] if a.path else [*sorted(NOTEBOOKS.glob("decisions*.ipynb")), NOTEBOOKS / "decision-plot.ipynb"]
    total = 0
    for path in paths:
        failed = run(path, a.timeout)
        total += len(failed)
        print(f"executed {path}; {len(failed)} failing claim check(s)")
        for line in failed:
            print(f"  {line}")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
