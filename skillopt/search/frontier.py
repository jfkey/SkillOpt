"""Pareto-frontier extraction + prior-based deployment selection.

The frontier is the *menu* (mcts_06 §2.2): the non-dominated set over the 2-D
absolute values ``(success, cost)`` of all evaluated nodes — maximize success,
minimize cost.  Frontier extraction is λ-free (Option B); λ-sweeps only refine
the menu.  Deploying one ``best_skill.md`` then means picking a point with a
*declared prior* (perf vs efficiency).
"""
from __future__ import annotations

from typing import Iterable, Optional

from skillopt.search.node import NodeValue, SkillNode


def dominates(a: NodeValue, b: NodeValue) -> bool:
    """``a`` Pareto-dominates ``b``: ≥ in success, ≤ in cost, strict in one."""
    not_worse = a.success >= b.success and a.cost <= b.cost
    strictly_better = a.success > b.success or a.cost < b.cost
    return not_worse and strictly_better


def pareto_front(nodes: Iterable[SkillNode]) -> list[SkillNode]:
    """Non-dominated subset of *evaluated* nodes (those with a value).

    Returns nodes sorted by ascending cost (then descending success), which is
    the natural left-to-right order along a success-cost frontier.  Ties on the
    exact ``(success, cost)`` point keep only the first (shallowest) node.
    """
    evaluated = [n for n in nodes if n.value is not None]
    front: list[SkillNode] = []
    for n in evaluated:
        if any(other is not n and dominates(other.value, n.value) for other in evaluated):
            continue
        # Drop exact-duplicate points (same success & cost) — keep the first.
        if any(
            f.value.success == n.value.success and f.value.cost == n.value.cost
            for f in front
        ):
            continue
        front.append(n)
    front.sort(key=lambda n: (n.value.cost, -n.value.success))
    return front


def select_by_prior(
    front: list[SkillNode],
    prior: str = "perf",
    efficiency_eps: float = 0.01,
) -> Optional[SkillNode]:
    """Collapse the frontier menu to one deployment point (mcts_06 §2.2).

    * ``perf``       — highest success (tie-break: lowest cost).
    * ``efficiency`` — lowest cost among nodes whose success is within
      ``efficiency_eps`` of the best success on the frontier.
    """
    if not front:
        return None
    best_success = max(n.value.success for n in front)
    if prior == "efficiency":
        eligible = [n for n in front if n.value.success >= best_success - efficiency_eps]
        return min(eligible, key=lambda n: (n.value.cost, -n.value.success))
    # default: perf
    return max(front, key=lambda n: (n.value.success, -n.value.cost))
