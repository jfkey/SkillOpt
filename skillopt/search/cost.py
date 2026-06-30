"""Skill-induced deployment cost (mcts_02 §1).

The second objective of the search.  Cost = the total token cost a skill
induces at deployment time: the resident skill tokens (the skill is prepended
to every task's context) plus any extra generation/turns the skill provokes.

For SearchQA the rollout records ``cost_total_tokens`` per item
(``prompt_tokens`` already absorbs the prepended skill); this module just reads
that channel and aggregates it.  Kept env-agnostic so other benchmarks can
populate the same key (or fall back to a char/4 proxy).
"""
from __future__ import annotations


def _get(result: object, key: str, default: float = 0.0) -> float:
    if isinstance(result, dict):
        val = result.get(key, default)
    else:
        val = getattr(result, key, default)
    try:
        return float(val if val is not None else default)
    except (TypeError, ValueError):
        return float(default)


def task_cost(result: object) -> float:
    """Per-task deployment cost (headline = total tokens).

    Falls back to ``prompt_tokens + completion_tokens`` and then to a char/4
    proxy of the response text when the explicit channel is absent (e.g. an
    env that has not yet been wired for the cost channel).
    """
    total = _get(result, "cost_total_tokens", 0.0)
    if total > 0:
        return total
    pt = _get(result, "prompt_tokens", 0.0)
    ct = _get(result, "completion_tokens", 0.0)
    if pt + ct > 0:
        return pt + ct
    # Last-resort proxy so cost is never silently zero.
    resp = result.get("response", "") if isinstance(result, dict) else getattr(result, "response", "")
    return float(len(resp or "")) / 4.0


def per_task_cost(results: list) -> dict[str, float]:
    """Map ``task_id -> cost`` for a rollout batch."""
    out: dict[str, float] = {}
    for r in results:
        rid = str(r.get("id") if isinstance(r, dict) else getattr(r, "id", ""))
        out[rid] = task_cost(r)
    return out


def mean_cost(results: list) -> float:
    """Mean per-task cost over a rollout batch (0.0 if empty)."""
    if not results:
        return 0.0
    return sum(task_cost(r) for r in results) / len(results)
