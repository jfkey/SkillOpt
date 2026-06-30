"""Cost-aware multi-objective MCTS over the skill-version tree.

A decoupled search layer on top of SkillOpt's existing stage functions
(rollout / reflect / aggregate / clip / apply).  Nodes are skill-document
versions; edges are budgeted edits; node value is the 2-D vector
``(success, cost)``.  UCT search produces a success-cost Pareto frontier
instead of a single ``best_skill.md``.

Design: see ``.record/mcts_00`` … ``mcts_06``.  The target and optimizer
models stay frozen throughout; deployment is still zero extra model calls.

Public entry point: :func:`skillopt.search.runner.run_mcts_search`.
"""
from __future__ import annotations

from skillopt.search.node import NodeValue, SkillNode, scalarize
from skillopt.search.frontier import dominates, pareto_front, select_by_prior

__all__ = [
    "NodeValue",
    "SkillNode",
    "scalarize",
    "dominates",
    "pareto_front",
    "select_by_prior",
]
