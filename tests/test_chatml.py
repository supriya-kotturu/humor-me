"""D-124: the prompt text training uses is the text Ollama's qwen3:4b-instruct template renders, offline.

`tests/fixtures/ollama_qwen3_4b_instruct.template` is the template blob read from the local Ollama store; `..._renders.json` is
what Go's text/template renders for four message lists (`render_ollama_template.go.txt` is the program, run once by hand, with
Ollama's own collation of system messages). The Hugging Face template is a transcribed fixture (the no-tools branches); the
Explorer smoke job compares the real tokenizer against `render_hf`."""
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from explorer import chatml  # noqa: E402

FIX = pathlib.Path(__file__).parent / "fixtures"
CASES = {
    "system_user": [{"role": "system", "content": "SYS line one.\nLine two."}, {"role": "user", "content": '{"venue": "bar"}'}],
    "user_only": [{"role": "user", "content": "hi"}],
    "system_user_assistant": [{"role": "system", "content": "SYS"}, {"role": "user", "content": "U1"}, {"role": "assistant", "content": "A1"}],
    "system_user_asst_user": [{"role": "system", "content": "SYS"}, {"role": "user", "content": "U1"},
                              {"role": "assistant", "content": "A1"}, {"role": "user", "content": "U2"}],
}
GOLDEN = json.loads((FIX / "ollama_qwen3_4b_instruct_renders.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", sorted(CASES))
def test_render_ollama_equals_the_real_templates_rendering(name):
    assert chatml.render_ollama(CASES[name]) == GOLDEN[name]


def test_the_fixture_is_the_template_the_golden_file_came_from():
    text = (FIX / "ollama_qwen3_4b_instruct.template").read_text(encoding="utf-8")
    assert "<|im_start|>system" in text and "{{ .System }}\n\n{{ end }}" in text  # the blank line after the system text


def test_the_hf_template_differs_from_ollamas_only_by_the_blank_line_after_the_system_text():
    jinja2 = pytest.importorskip("jinja2")
    template = jinja2.Environment().from_string((FIX / "qwen3_4b_instruct_2507_chat_template.jinja").read_text(encoding="utf-8"))
    for name in ("system_user", "user_only", "system_user_asst_user"):
        msgs = CASES[name]
        hf = template.render(messages=msgs, add_generation_prompt=True)
        assert hf.strip() == chatml.render_hf(msgs).strip()
        ollama = chatml.render_ollama(msgs)
        assert hf != ollama if msgs[0]["role"] == "system" else hf == ollama
        assert hf == ollama.replace("\n\n<|im_end|>\n<|im_start|>user", "<|im_end|>\n<|im_start|>user", 1)


def test_the_reply_is_trained_after_the_open_assistant_turn():
    msgs = CASES["system_user"]
    prompt = chatml.render_ollama(msgs)
    assert prompt.endswith("<|im_start|>assistant\n")
    # prompt + reply + end token is what Ollama would render for the history with the reply, before its own closing
    assert chatml.render_ollama(msgs + [{"role": "assistant", "content": "REPLY"}]) == prompt + "REPLY"


def test_consecutive_messages_of_a_role_merge_like_ollamas():
    got = chatml.render_ollama([{"role": "system", "content": "A"}, {"role": "system", "content": "B"}, {"role": "user", "content": "U"}])
    assert got.startswith("<|im_start|>system\nA\n\nB\n\n<|im_end|>\n")
