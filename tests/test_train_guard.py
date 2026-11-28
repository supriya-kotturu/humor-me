"""The training jobs' NaN stop rule, tokenizer source, token-id and end-token checks (explorer/train_common.py), with fakes:
no torch, no model, no network."""
import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from explorer import train_common as tcm  # noqa: E402
from explorer.chatml import IM_END  # noqa: E402

NAN, INF = float("nan"), float("inf")


def feed(rule, steps):
    return [rule.update(loss, grad) for loss, grad in steps]


@pytest.mark.parametrize("skipped", [1, 2, 3, 4])
def test_a_few_skipped_fp16_steps_with_a_finite_loss_do_not_stop(skipped):
    """The loss scaler's first-step overflow (smoke run, 2026-11-27: loss ln 2, grad_norm nan, learning rate 0) must not stop."""
    rule = tcm.NanRule()
    out = feed(rule, [(math.log(2), NAN)] * skipped + [(0.69, 0.9)] * 3)
    assert not any(stop for stop, _ in out)
    assert all("skipped" in msg for _, msg in out[:skipped]) and all(msg is None for _, msg in out[skipped:])
    assert rule.run == 0  # a finite grad norm resets the count


def test_five_skipped_steps_in_a_row_stop():
    rule = tcm.NanRule()
    out = feed(rule, [(0.69, NAN), (0.69, INF), (0.69, NAN), (0.69, NAN), (0.69, NAN)])
    assert [stop for stop, _ in out] == [False] * 4 + [True]
    assert "5 logged steps in a row" in out[-1][1]


def test_skips_that_are_not_in_a_row_never_add_up_to_a_stop():
    rule = tcm.NanRule()
    out = feed(rule, [(0.69, NAN)] * 4 + [(0.69, 1.0)] + [(0.69, NAN)] * 4)
    assert not any(stop for stop, _ in out)


@pytest.mark.parametrize("loss", [NAN, INF, -INF])
def test_a_non_finite_loss_stops_at_once(loss):
    stop, msg = tcm.NanRule().update(loss, 0.5)
    assert stop and "loss is" in msg


def test_plain_fp32_stops_on_the_first_nan_grad_norm_since_no_scaler_skips_it():
    stop, _ = tcm.NanRule(k=1).update(0.69, NAN)
    assert stop


def test_a_log_without_loss_or_grad_norm_changes_nothing():
    rule = tcm.NanRule()
    rule.update(0.69, NAN)
    assert rule.update(None, None) == (False, None) and rule.run == 1


def test_only_fp16_uses_mixed_precision_and_the_advice_names_the_fallback():
    assert tcm.amp("fp16") and not tcm.amp("fp32")
    assert "DTYPE=fp32" in tcm.unstable_advice("fp16") and "DTYPE=fp32" not in tcm.unstable_advice("fp32")


def test_the_tokenizer_is_always_the_base_models_never_a_merged_folders(tmp_path):
    merged = tmp_path / "smoke-merged"
    merged.mkdir()
    assert tcm.tokenizer_source(str(merged)) == tcm.DEFAULT_BASE
    assert tcm.tokenizer_source(tcm.DEFAULT_BASE) == tcm.DEFAULT_BASE


class FakeTok:
    """One id per character, shifted by `offset`; eos and the ChatML end token as given."""

    def __init__(self, offset=0, eos=IM_END):
        self.offset, self.eos_token = offset, eos

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) + self.offset for c in text]}

    def convert_tokens_to_ids(self, token):
        return 7 if token == IM_END else 0

    def decode(self, ids):
        return "".join("E" if i == 7 else "x" for i in ids)


def test_same_tokens_passes_equal_ids_and_refuses_different_ones():
    tcm.same_tokens(FakeTok(), FakeTok(), "merged folder")
    with pytest.raises(SystemExit, match="differently from the base"):
        tcm.same_tokens(FakeTok(offset=1), FakeTok(), "merged folder")


def test_the_sample_covers_what_a_changed_regex_would_split_differently():
    for piece in ("I'm", "can't", "12345", "café", "  two", "\t", "<|im_end|>"):
        assert piece in tcm.TOKEN_SAMPLE


def test_the_end_token_must_be_im_end():
    tcm.end_token_check(FakeTok())
    with pytest.raises(SystemExit, match="eos_token"):
        tcm.end_token_check(FakeTok(eos="<|endoftext|>"))


def test_a_reply_must_end_in_exactly_one_end_token():
    tok = FakeTok()
    tcm.inspect_tokens(tok, [1, 2, 7], "chosen_input_ids")
    tcm.inspect_tokens(tok, [1, 2], "prompt_input_ids", ends=False)
    for ids in ([1, 2], [1, 7, 7], []):
        with pytest.raises(SystemExit, match="exactly one"):
            tcm.inspect_tokens(tok, ids, "chosen_input_ids")


def test_the_callback_logs_scaler_skips_and_stops_only_after_k_in_a_row(capsys):
    pytest.importorskip("transformers")
    from types import SimpleNamespace

    from transformers import TrainerControl

    guard, control = tcm.callbacks(max_hours=1.0, dtype="fp16"), TrainerControl()
    scaler = SimpleNamespace(get_scale=lambda: 32768.0)
    opt = SimpleNamespace(step_was_skipped=True, scaler=scaler)
    for step in range(1, tcm.NAN_STEPS):
        state = SimpleNamespace(global_step=step)
        guard.on_step_end(None, state, control, optimizer=opt)
        guard.on_log(None, state, control, logs={"loss": 0.6931, "grad_norm": NAN, "learning_rate": 0.0})
        assert not control.should_training_stop and not guard.nan
    out = capsys.readouterr().out
    assert "[scaler] step 1 skipped (fp16 gradient overflow); loss scale now 32768.0" in out and "[stop]" not in out
    guard.on_log(None, SimpleNamespace(global_step=tcm.NAN_STEPS), control, logs={"loss": 0.6931, "grad_norm": NAN})
    assert control.should_training_stop and guard.nan
    assert "fp16 unstable, rerun with DTYPE=fp32" in capsys.readouterr().out


def test_the_end_of_training_summary_log_is_ignored():
    pytest.importorskip("transformers")
    from types import SimpleNamespace

    from transformers import TrainerControl

    guard, control = tcm.callbacks(max_hours=1.0), TrainerControl()
    guard.on_log(None, SimpleNamespace(global_step=10), control, logs={"train_loss": NAN, "train_runtime": 61.0})
    assert not control.should_training_stop and guard.rule.run == 0
