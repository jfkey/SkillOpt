"""Tree node + 2-D node value for the skill-version MCTS (mcts_01 §1).

Each node is one skill-document version (state = ``skill_hash``).  Value is the
2-D vector ``(success, cost)`` carried as an absolute, telescoping-anchored
quantity.  Backpropagation accumulates the 2-D vector ``W`` and visit count
``N`` (Option B: λ is *not* baked in at backup); the navigation scalar ``Q`` is
computed on demand at selection time via :func:`scalarize`, so a single tree
serves every λ and frontier extraction is λ-free (mcts_01 §5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class NodeValue:
    """A 2-D node value: success (maximize) and cost (minimize)."""

    success: float
    cost: float

    def __add__(self, other: "NodeValue") -> "NodeValue":
        return NodeValue(self.success + other.success, self.cost + other.cost)

    def __sub__(self, other: "NodeValue") -> "NodeValue":
        return NodeValue(self.success - other.success, self.cost - other.cost)


def scalarize(value: NodeValue, lam: float, cost_ref: float) -> float:
    """Scalarize a 2-D value for UCT navigation / SH ranking (mcts_02 §6).

    ``scal_λ(v) = v.success − λ · max(0, ĉ − 1)``, where ``ĉ = cost / cost_ref``
    is the cost relative to the no-skill baseline.  Only cost *above* the
    baseline is penalized, so a skill that is cheaper than baseline is never
    pushed to shrink further.  λ appears *only* here (not at backup).
    """
    if cost_ref and cost_ref > 0:
        c_hat = value.cost / cost_ref
    else:
        c_hat = 1.0
    return value.success - lam * max(0.0, c_hat - 1.0)


@dataclass
class SkillNode:
    """A node in the skill-version tree."""

    skill: str
    skill_hash: str
    depth: int = 0
    node_id: int = 0
    parent: Optional["SkillNode"] = None
    children: list["SkillNode"] = field(default_factory=list)

    # Edge: the selected edits that produced this node from its parent.
    edge_edits: Optional[dict] = None

    # 2-D absolute value (telescoping-anchored: v = parent.value + Δ).
    value: Optional[NodeValue] = None

    # Backprop accumulators (Option B: 2-D vector, λ not baked in).
    W_success: float = 0.0
    W_cost: float = 0.0
    N: int = 0

    # Per-task scores on the selection subset: task_id -> (success, cost).
    # Supports CRN-paired comparison and successive-halving reuse (milestone 2).
    eval_tasks: dict[str, tuple[float, float]] = field(default_factory=dict)

    # Cached train-rollout evidence for expansion (rolled once, reused).
    train_evidence: Optional[list] = None
    train_rollout_dir: Optional[str] = None

    expanded: bool = False
    terminal: bool = False

    def q(self, lam: float, cost_ref: float) -> float:
        """Mean-value navigation scalar ``scal_λ(W / N)`` (∞ if unvisited)."""
        if self.N == 0:
            return float("inf")
        mean = NodeValue(self.W_success / self.N, self.W_cost / self.N)
        return scalarize(mean, lam, cost_ref)

    def to_record(self) -> dict:
        """Serializable summary for ``tree.json``."""
        return {
            "node_id": self.node_id,
            "parent_id": self.parent.node_id if self.parent else None,
            "depth": self.depth,
            "skill_hash": self.skill_hash,
            "skill_len": len(self.skill),
            "value": None if self.value is None else {
                "success": self.value.success,
                "cost": self.value.cost,
            },
            "N": self.N,
            "W": {"success": self.W_success, "cost": self.W_cost},
            "expanded": self.expanded,
            "terminal": self.terminal,
            "n_children": len(self.children),
        }
