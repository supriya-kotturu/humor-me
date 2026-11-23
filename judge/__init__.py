"""JUDGE: rank one prompt's candidate jokes from pairwise "which is funnier?" calls
(Bradley-Terry + adaptive LUCB pair selection). See judge/README.md.

    bt.py       Bradley-Terry math: fit, standard errors, P(i beats j), Borda
    judges.py   SimulatedJudge (tests) and LLMJudge (llama3.1:8b on Ollama)
    select.py   which pairs to ask: all pairs, warm start, adaptive LUCB
    rank.py     rank(): ask the judge, fit, stop; write_judgments()
    records.py  Comparison (one judge call) and Ranking (one prompt's result)
    run_llm.py  command line: rank a joke list with the LLM judge
"""
from .bt import fit_bt, win_prob  # NEW
from .judges import LLMJudge, SimulatedJudge  # NEW
from .rank import rank, write_judgments
from .records import Comparison, Ranking  # NEW

__all__ = ["rank", "Ranking", "Comparison", "SimulatedJudge", "LLMJudge", "fit_bt", "win_prob",  # NEW
           "write_judgments"]  # NEW
