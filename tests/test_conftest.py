import os

from tests.conftest import restored_environ


def test_restored_environ_undoes_sets_changes_and_deletes(monkeypatch):
    monkeypatch.setenv("HUMOR_KEEP", "before")
    monkeypatch.setenv("HUMOR_GONE", "before")
    monkeypatch.delenv("HUMOR_NEW", raising=False)
    with restored_environ():
        os.environ["HUMOR_NEW"] = "leaked"
        os.environ.setdefault("JOKE_WRITER_PROVIDER_TEST", "ollama")
        os.environ["HUMOR_KEEP"] = "changed"
        del os.environ["HUMOR_GONE"]
    assert "HUMOR_NEW" not in os.environ and "JOKE_WRITER_PROVIDER_TEST" not in os.environ
    assert os.environ["HUMOR_KEEP"] == "before" and os.environ["HUMOR_GONE"] == "before"


def test_restored_environ_restores_after_an_exception():
    try:
        with restored_environ():
            os.environ["HUMOR_RAISED"] = "x"
            raise RuntimeError
    except RuntimeError:
        pass
    assert "HUMOR_RAISED" not in os.environ
