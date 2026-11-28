"""Shared test setup."""
import contextlib
import os

import pytest


@contextlib.contextmanager
def restored_environ():
    """Puts os.environ back as it was on exit. Several command-line `main()`s set variables straight on os.environ
    (sim.dpo_pairs sets JOKE_WRITER_PROVIDER=ollama; --env-file loads a .env), and a test that calls one used to
    leave them set, so a later test passed or failed by suite order (dev-log, PR #90)."""
    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


@pytest.fixture(autouse=True)
def _isolate_environ():
    with restored_environ():
        yield
