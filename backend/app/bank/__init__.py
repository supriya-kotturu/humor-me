"""Hand-written joke bank: material that needs no model, for tests, calibration and outages."""
import pathlib

import yaml

BANK_PATH = pathlib.Path(__file__).with_name("bank-v1.yaml")


def load_bank(path: pathlib.Path = BANK_PATH) -> dict[str, list[dict]]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class BankGenerator:
    """One instance per show: bit-to-slot usage lives on it, and a reused instance would carry one show's history into the next."""

    generator_id = "bank-v1"

    def __init__(self, bank: dict[str, list[dict]] | None = None):
        self.bank = bank if bank is not None else load_bank()
        self._used: dict[str, list[int]] = {m: [0] * len(bits) for m, bits in self.bank.items()}
        self._assigned: dict[int, tuple[str, int]] = {}

    def candidates(self, mode: str, bit: int, joke: int) -> list[str]:
        if mode not in self.bank:
            raise KeyError(f"unknown mode: {mode}")
        slot = self._slot(mode, bit)
        jokes = self.bank[mode][slot]["jokes"]
        return [jokes[joke]["text"]]

    def _slot(self, mode: str, bit: int) -> int:
        held = self._assigned.get(bit)
        if held is not None and held[0] == mode:
            return held[1]
        # Ties go to the lowest index, so a show with the same modes always replays the same lines.
        counts = self._used[mode]
        slot = counts.index(min(counts))
        counts[slot] += 1
        self._assigned[bit] = (mode, slot)
        return slot
