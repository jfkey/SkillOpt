"""Node value estimation (mcts_01 §4, mcts_02 §3-5).

A node's 2-D value ``(success, cost)`` is its skill's evaluation on a *fixed,
ordered* selection subset ``S``.  The subset is fixed so that (a) the per-skill
rollout cache is a transposition table keyed by ``skill_hash`` and (b) every
node is compared on the *same* tasks — the CRN anchor.

Two estimators share one interface (``anchor`` + ``estimate_children`` +
``eval_content``) so the search loop is agnostic to which is used:

* :class:`ValueEstimator` — **full mode**: every node is evaluated on the whole
  subset, value is absolute, telescoping is trivially exact.  This is the
  ablation baseline (CRN/SH off) and the test-split estimator.
* :class:`PairedValueEstimator` — **paired mode (M2)**: CRN-paired Δ vs the
  parent + successive halving across the ``b`` siblings of one expansion.  The
  parent is anchored to full coverage once and reused as the CRN reference;
  cheap-looking children are pruned early so they never spend the full budget.

Both track ``n_evals`` (candidate children evaluated — the search-budget unit)
and ``n_task_rollouts`` (total per-(skill, task) evaluations — the compute /
overfitting unit, reported but not budget-capped).
"""
from __future__ import annotations

import math
import os

from skillopt.evaluation.gate import select_gate_score
from skillopt.search.cost import task_cost
from skillopt.search.node import NodeValue, SkillNode


class ValueEstimator:
    """Full-mode estimator: evaluate each node on the whole fixed subset."""

    def __init__(
        self,
        adapter,
        sel_items: list,
        *,
        out_root: str,
        gate_metric: str = "hard",
        mixed_weight: float = 0.5,
        cache: dict | None = None,
    ) -> None:
        self.adapter = adapter
        self.sel_items = sel_items
        self.task_ids = [str(it.get("id") if isinstance(it, dict) else getattr(it, "id", idx))
                         for idx, it in enumerate(sel_items)]
        self.out_root = out_root
        self.gate_metric = gate_metric
        self.mixed_weight = mixed_weight
        # hash -> {"per_task": {task_id: (succ, cost)}, "k": covered_prefix_len}
        self.cache: dict[str, dict] = cache if cache is not None else {}
        self.n_evals = 0          # candidate (child) evaluations — budget unit
        self.n_task_rollouts = 0  # per-(skill, task) evaluations — compute unit

    # ── rollout core (prefix-cached, resume-aware) ─────────────────────────
    def _score(self, r) -> float:
        hard = float(r.get("hard", 0) if isinstance(r, dict) else getattr(r, "hard", 0))
        soft = float(r.get("soft", 0.0) if isinstance(r, dict) else getattr(r, "soft", 0.0))
        return select_gate_score(hard, soft, self.gate_metric, self.mixed_weight)

    def _eval_prefix(self, skill_hash_: str, skill: str, m: int) -> dict:
        """Evaluate ``skill`` on the first ``m`` selection tasks.

        Prefix-cached: coverage only ever grows, and the underlying rollout is
        resume-aware, so the marginal cost is just the newly-added tasks.
        """
        m = max(0, min(int(m), len(self.sel_items)))
        rec = self.cache.setdefault(skill_hash_, {"per_task": {}, "k": 0})
        if rec["k"] >= m:
            return rec
        tasks = self.sel_items[:m]
        out_dir = os.path.join(self.out_root, "sel_eval", skill_hash_)
        results = self.adapter.rollout(tasks, skill, out_dir)
        for r in results:
            rid = str(r.get("id") if isinstance(r, dict) else getattr(r, "id", ""))
            rec["per_task"][rid] = (self._score(r), task_cost(r))
        new = m - rec["k"]
        if new > 0:
            self.n_task_rollouts += new
        rec["k"] = max(rec["k"], m)
        return rec

    def _mean_value(self, rec: dict, m: int) -> NodeValue:
        ids = [i for i in self.task_ids[:m] if i in rec["per_task"]]
        n = max(len(ids), 1)
        succ = sum(rec["per_task"][i][0] for i in ids) / n
        cost = sum(rec["per_task"][i][1] for i in ids) / n
        return NodeValue(succ, cost)

    # ── shared interface ───────────────────────────────────────────────────
    def anchor(self, node: SkillNode) -> NodeValue:
        """Full-coverage absolute evaluation (exact value + per-task for CRN)."""
        m = len(self.sel_items)
        rec = self._eval_prefix(node.skill_hash, node.skill, m)
        node.eval_tasks = {i: rec["per_task"][i] for i in self.task_ids[:m] if i in rec["per_task"]}
        node.value = self._mean_value(rec, m)
        return node.value

    def estimate_children(self, children, parent, *, lambda_cost: float = 0.0, cost_ref: float = 1.0):
        """Full mode: each child fully + independently evaluated (no SH/CRN)."""
        for ch in children:
            self.anchor(ch)
            self.n_evals += 1

    def eval_content(self, skill_hash_: str, skill: str) -> dict:
        """Evaluate a skill on the full subset; return success/cost (test eval)."""
        m = len(self.sel_items)
        rec = self._eval_prefix(skill_hash_, skill, m)
        mv = self._mean_value(rec, m)
        return {"success": mv.success, "cost": mv.cost, "n": m}

    # Back-compat alias (older callers / mocks may use evaluate()).
    def evaluate(self, node: SkillNode) -> NodeValue:
        return self.anchor(node)


class PairedValueEstimator(ValueEstimator):
    """Paired mode (M2): CRN-paired Δ + successive halving over the b siblings.

    The parent is anchored to full coverage once (the CRN reference); each child
    is compared to it on a *shared, fixed* task prefix, so the paired Δ cancels
    the dominant task-to-task variance.  Successive halving evaluates all
    siblings on a small prefix ``n0``, keeps the top ``1/η`` by scalarized
    paired Δ, grows the prefix ×η, and repeats — so weak children never pay for
    the full subset.  Telescoping anchors each child to the parent:
    ``child.value = parent.value + Δ`` (exact for the SH winner at full
    coverage; an approximation for early-pruned children).
    """

    def __init__(self, *args, n0: int = 4, sh_eta: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n0 = max(1, int(n0))
        self.sh_eta = max(2, int(sh_eta))

    def _paired_delta(self, parent: SkillNode, child: SkillNode, m: int) -> tuple[float, float]:
        """Mean (Δsucc, Δcost) over the shared first-``m`` tasks (CRN)."""
        prec = self.cache.get(parent.skill_hash, {}).get("per_task", {})
        crec = self.cache.get(child.skill_hash, {}).get("per_task", {})
        ds = dc = 0.0
        n = 0
        for i in self.task_ids[:m]:
            if i in prec and i in crec:
                ds += crec[i][0] - prec[i][0]
                dc += crec[i][1] - prec[i][1]
                n += 1
        n = max(n, 1)
        return ds / n, dc / n

    def estimate_children(self, children, parent, *, lambda_cost: float = 0.0, cost_ref: float = 1.0):
        if not children:
            return
        s_full = len(self.sel_items)
        # The parent is the CRN reference — guarantee full per-task coverage.
        if parent.value is None or self.cache.get(parent.skill_hash, {}).get("k", 0) < s_full:
            self.anchor(parent)

        ref = cost_ref or 1.0
        alive = list(children)
        delta: dict[int, tuple[float, float, int]] = {}  # node_id -> (ds, dc, prefix_m)
        m = min(self.n0, s_full)

        while True:
            for ch in alive:
                self._eval_prefix(ch.skill_hash, ch.skill, m)
                ds, dc = self._paired_delta(parent, ch, m)
                delta[ch.node_id] = (ds, dc, m)
            if len(alive) <= 1 or m >= s_full:
                break
            # Successive halving: rank by scalarized paired Δ (favour success
            # gained per unit of cost added), keep the top 1/η.
            alive.sort(
                key=lambda c: delta[c.node_id][0] - lambda_cost * (delta[c.node_id][1] / ref),
                reverse=True,
            )
            keep = max(1, math.ceil(len(alive) / self.sh_eta))
            alive = alive[:keep]
            m = min(m * self.sh_eta, s_full)

        # Promote survivors to full coverage so the winner's telescoped value is
        # exact (parent is anchored on all of S, so Δ on full S ⇒ exact child).
        for ch in alive:
            if delta[ch.node_id][2] < s_full:
                self._eval_prefix(ch.skill_hash, ch.skill, s_full)
                ds, dc = self._paired_delta(parent, ch, s_full)
                delta[ch.node_id] = (ds, dc, s_full)

        # Telescoping anchor for every child on its largest evaluated prefix.
        for ch in children:
            ds, dc, mm = delta[ch.node_id]
            ch.value = NodeValue(parent.value.success + ds, parent.value.cost + dc)
            rec = self.cache.get(ch.skill_hash, {"per_task": {}})
            ch.eval_tasks = {i: rec["per_task"][i] for i in self.task_ids[:mm] if i in rec["per_task"]}
            self.n_evals += 1
