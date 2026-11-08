"""Backend package. sim/ sits at the repo root, so put it on the path here: every app module can import it,
whichever is imported first, and uvicorn can run from backend/."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
