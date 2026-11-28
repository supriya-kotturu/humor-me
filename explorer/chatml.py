"""The prompt text a Qwen3-4B-Instruct-2507 writer is trained on: Ollama's rendering, not Hugging Face's (D-124).

The show talks to the model through Ollama, whose `qwen3:4b-instruct` template puts a blank line after the system text
(`<|im_start|>system\\nSYS\\n\\n<|im_end|>`); the Hugging Face template does not (`SYS<|im_end|>`). The pool was written, and the
trained writer will be served, through Ollama's, so training on any other text would show the model a prompt it never sees live.
`render_ollama` is checked against Go's text/template rendering of the real template blob (tests/fixtures), and the smoke job
checks `render_hf` against the tokenizer on the cluster. No dependencies: it runs on the laptop and in the cluster venv.
"""
IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def collate(messages: list[dict]) -> tuple[str, list[dict]]:
    """Ollama's own preparation: the system text is every system message joined by a blank line; consecutive messages of one
    role merge the same way."""
    system, out = [], []
    for m in messages:
        if m["role"] == "system":
            system.append(m["content"])
        if out and out[-1]["role"] == m["role"]:
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + "\n\n" + m["content"]}
        else:
            out.append(dict(m))
    return "\n\n".join(system), out


def render_ollama(messages: list[dict]) -> str:
    """What Ollama's qwen3:4b-instruct template renders for messages without tools. The last assistant message, if any, is left
    open (no `<|im_end|>`), and a last user message gets the generation prompt."""
    system, msgs = collate(messages)
    text = f"{IM_START}system\n{system}\n\n{IM_END}\n" if system else ""
    for i, m in enumerate(msgs):
        last = i == len(msgs) - 1
        if m["role"] == "user":
            text += f"{IM_START}user\n{m['content']}{IM_END}\n"
        elif m["role"] == "assistant":
            text += f"{IM_START}assistant\n{m['content']}" + ("" if last else f"{IM_END}\n")
        if m["role"] != "assistant" and last:
            text += f"{IM_START}assistant\n"
    return text


def render_hf(messages: list[dict], generation_prompt: bool = True) -> str:
    """What the Hugging Face template renders for the same messages (system first, no tools): the same text except that the
    system block has no blank line before `<|im_end|>`. The smoke job asserts the tokenizer agrees with this."""
    text = ""
    for m in messages:
        text += f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n"
    return text + (f"{IM_START}assistant\n" if generation_prompt else "")
