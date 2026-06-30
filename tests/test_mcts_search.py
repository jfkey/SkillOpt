"""Tests for the cost-aware MCTS skill search (skillopt.search).

All offline — no model API.  Covers the pure 2-D math (scalarize / dominate /
frontier / prior selection) and a full MCTS run driven by mock world-model and
value-estimator stand-ins, so select / expand / 2-D backup / termination /
frontier extraction are all exercised end-to-end.
"""
from __future__ import annotations

import math

import pytest

from skillopt.search.cost import mean_cost, task_cost
from skillopt.search.frontier import dominates, pareto_front, select_by_prior
from skillopt.search.mcts import MCTS
from skillopt.search.node import NodeValue, SkillNode, scalarize
from skillopt.search.value import ValueEstimator
from skillopt.search.world_model import TransitionResult, normalise_patches
from skillopt.utils import skill_hash


# ── pure 2-D value math ──────────────────────────────────────────────────────

def test_scalarize_only_penalizes_above_baseline():
    cost_ref = 100.0
    # cheaper than baseline → no penalty regardless of λ
    cheap = NodeValue(success=0.6, cost=80.0)
    assert scalarize(cheap, lam=1.0, cost_ref=cost_ref) == pytest.approx(0.6)
    # 2× baseline cost, λ=0.3 → 0.6 - 0.3*(2-1) = 0.3
    pricey = NodeValue(success=0.6, cost=200.0)
    assert scalarize(pricey, lam=0.3, cost_ref=cost_ref) == pytest.approx(0.3)
    # λ=0 recovers pure success
    assert scalarize(pricey, lam=0.0, cost_ref=cost_ref) == pytest.approx(0.6)


def test_node_value_add_sub_telescoping():
    parent = NodeValue(0.5, 100.0)
    delta = NodeValue(0.1, 20.0)
    child = parent + delta
    assert child.success == pytest.approx(0.6)
    assert child.cost == pytest.approx(120.0)
    assert (child - parent).success == pytest.approx(0.1)


def test_dominates():
    a = NodeValue(0.8, 100.0)
    b = NodeValue(0.7, 120.0)  # a is better on both
    assert dominates(a, b)
    assert not dominates(b, a)
    # equal point does not dominate itself
    assert not dominates(a, NodeValue(0.8, 100.0))
    # trade-off: neither dominates
    c = NodeValue(0.9, 200.0)
    assert not dominates(a, c)
    assert not dominates(c, a)


def _node(success, cost, nid=0):
    n = SkillNode(skill=f"s{nid}", skill_hash=f"h{nid}", node_id=nid)
    n.value = NodeValue(success, cost)
    return n


def test_pareto_front_and_prior_selection():
    nodes = [
        _node(0.50, 100.0, 0),   # cheap, low success — on frontier
        _node(0.70, 150.0, 1),   # mid — on frontier
        _node(0.90, 300.0, 2),   # best success — on frontier
        _node(0.60, 160.0, 3),   # dominated by node 1
    ]
    front = pareto_front(nodes)
    ids = sorted(n.node_id for n in front)
    assert ids == [0, 1, 2]            # node 3 dominated out
    # sorted ascending cost
    assert [n.value.cost for n in front] == [100.0, 150.0, 300.0]

    perf = select_by_prior(front, "perf")
    assert perf.node_id == 2          # highest success
    # No slack ⇒ only the top-success node qualifies ⇒ same as perf.
    eff0 = select_by_prior(front, "efficiency", efficiency_eps=0.0)
    assert eff0.node_id == 2
    # Generous slack ⇒ every frontier point qualifies ⇒ pick the cheapest.
    eff_all = select_by_prior(front, "efficiency", efficiency_eps=1.0)
    assert eff_all.node_id == 0
    # Mid slack: success within 0.25 of best (0.90) keeps nodes 1 (0.70) & 2;
    # node 0 (0.50) drops out ⇒ cheapest eligible is node 1.
    eff_mid = select_by_prior(front, "efficiency", efficiency_eps=0.25)
    assert eff_mid.node_id == 1


# ── cost channel ─────────────────────────────────────────────────────────────

def test_task_cost_prefers_explicit_then_falls_back():
    assert task_cost({"cost_total_tokens": 1234}) == 1234.0
    assert task_cost({"prompt_tokens": 100, "completion_tokens": 50}) == 150.0
    assert task_cost({"response": "x" * 40}) == pytest.approx(10.0)  # char/4
    assert mean_cost([{"cost_total_tokens": 100}, {"cost_total_tokens": 200}]) == 150.0


# ── normalise_patches ────────────────────────────────────────────────────────

def test_normalise_patches_splits_failure_success():
    raw = [
        {"source_type": "failure", "batch_size": 3, "patch": {"edits": [{"op": "add", "content": "a"}]}},
        {"source_type": "success", "batch_size": 2, "patch": {"edits": [{"op": "add", "content": "b"}]}},
        {"source_type": "failure", "patch": {"edits": []}},  # empty → dropped
        None,                                                 # non-dict → dropped
    ]
    failure, success = normalise_patches(raw, update_mode="patch")
    assert len(failure) == 1 and len(success) == 1
    # support_count + source_type stamped onto items
    assert failure[0]["edits"][0]["support_count"] == 3
    assert success[0]["edits"][0]["source_type"] == "success"


# ── full MCTS run with mocks (no API) ────────────────────────────────────────

class _MockWorldModel:
    """Deterministic edits: each child appends a distinct rule line."""

    def rollout_train_evidence(self, skill, train_items, out_dir):
        return []  # reflect evidence is irrelevant to the mock transition

    def transition(self, parent_skill, train_results, *, train_rollout_dir, expand_dir, seed):
        n_rules = parent_skill.count("- rule")
        candidate = parent_skill + f"\n- rule d{n_rules}s{seed}"
        return TransitionResult(candidate, {"edits": [{"op": "add"}]}, [])


class _MockValueEstimator:
    """Synthetic 2-D value: more rules → higher (saturating) success, higher cost.

    Creates a genuine success-cost trade-off so the frontier has > 1 point.
    Implements the estimator interface MCTS uses (anchor + estimate_children).
    """

    def __init__(self):
        self.cache: dict = {}
        self.n_evals = 0          # candidate children evaluated (budget unit)
        self.n_task_rollouts = 0

    def _value(self, skill):
        n = skill.count("- rule")
        return NodeValue(min(1.0, 0.5 + 0.1 * n), 100.0 + 40.0 * n)

    def anchor(self, node):
        node.value = self._value(node.skill)
        node.eval_tasks = {}
        return node.value

    def estimate_children(self, children, parent, *, lambda_cost=0.0, cost_ref=1.0):
        for ch in children:
            self.anchor(ch)
            self.n_evals += 1

    def eval_content(self, skill_hash_, skill):
        v = self._value(skill)
        return {"success": v.success, "cost": v.cost, "n": 1}


def _run(tmp_path, lambda_cost=0.0, branch=2, depth=3, budget=20):
    mcts = MCTS(
        _MockWorldModel(),
        _MockValueEstimator(),
        out_root=str(tmp_path),
        train_items=[],
        branch=branch,
        depth_limit=depth,
        c_explore=1.5,
        lambda_cost=lambda_cost,
        budget=budget,
        patience=10,
        max_iters=50,
        expand_seed=0,
    )
    return mcts, mcts.run("# skill\n")


def test_mcts_run_produces_nondominated_frontier(tmp_path):
    mcts, (root, evaluated, front) = _run(tmp_path)
    # root is always evaluated and anchors cost_ref
    assert root.value is not None
    assert mcts.cost_ref == pytest.approx(root.value.cost)
    # budget respected (soft cap: an in-flight expansion group of <=branch
    # children completes before the loop re-checks the budget)
    assert mcts.value_estimator.n_evals <= 20 + 2
    # frontier is genuinely non-dominated
    for a in front:
        for b in front:
            if a is not b:
                assert not dominates(a.value, b.value)
    # trade-off exists: cheapest (root, fewest rules) and a higher-success node
    assert len(front) >= 2
    costs = [n.value.cost for n in front]
    assert costs == sorted(costs)               # ascending cost order
    assert front[0].value.success <= front[-1].value.success


def test_mcts_backup_accumulates_2d_vector(tmp_path):
    mcts, (root, evaluated, front) = _run(tmp_path, budget=12)
    # Every ancestor's W is the sum of its subtree's child values; N matches.
    for node in mcts.nodes:
        if node.N == 0:
            continue
        # W/N must scalarize finite (sanity on Option-B accumulation)
        q = node.q(lam=0.3, cost_ref=mcts.cost_ref)
        assert math.isfinite(q)
    # root visited at least as many times as any descendant
    assert root.N >= max(n.N for n in mcts.nodes)


def test_mcts_terminates_and_respects_depth(tmp_path):
    mcts, (root, evaluated, front) = _run(tmp_path, depth=2, budget=100)
    assert all(n.depth <= 2 for n in mcts.nodes)
    # search halts (does not spin to max_iters worth of empty work forever)
    assert mcts.value_estimator.n_evals >= 1


def test_mcts_tree_grows_beyond_first_level(tmp_path):
    # Regression: SELECT must be able to descend, i.e. expand() attaches
    # children to node.children, so the tree reaches the depth limit.
    mcts, (root, evaluated, front) = _run(tmp_path, branch=2, depth=3, budget=30)
    assert root.children, "root must have attached children"
    assert max(n.depth for n in mcts.nodes) >= 2, "search never went past depth 1"
    # a node at depth >= 1 must itself have been expanded (has children)
    assert any(n.depth >= 1 and n.children for n in mcts.nodes)


# ── full runner orchestration with a mock adapter (no API) ───────────────────

class _MockAdapter:
    """Synthetic SearchQA-shaped adapter: success & cost both rise with rules."""

    def setup(self, cfg):
        self._cfg = cfg

    def build_eval_env(self, env_num, split, seed, **kw):
        n = env_num or 4
        return [{"id": f"{split}-{i}"} for i in range(n)]

    def build_train_env(self, batch_size, seed, **kw):
        return [{"id": f"train-{i}"} for i in range(batch_size)]

    def rollout(self, items, skill, out_dir, **kw):
        n_rules = skill.count("- rule")
        results = []
        for i, it in enumerate(items):
            hard = 1 if (i % 10) < (4 + 2 * n_rules) else 0   # more rules → more correct
            results.append({
                "id": it["id"],
                "hard": hard,
                "soft": min(1.0, 0.5 + 0.1 * n_rules),
                "cost_total_tokens": 100 + 40 * n_rules,      # more rules → costlier
            })
        return results

    def reflect(self, *a, **k):  # unused: world model is mocked
        return []


class _RunnerMockWorldModel:
    def __init__(self, adapter, **kw):
        self.adapter = adapter

    def rollout_train_evidence(self, skill, train_items, out_dir):
        return self.adapter.rollout(train_items, skill, out_dir)

    def transition(self, parent_skill, train_results, *, train_rollout_dir, expand_dir, seed):
        n = parent_skill.count("- rule")
        return TransitionResult(parent_skill + f"\n- rule d{n}s{seed}", {"edits": [{"op": "add"}]}, [])


def test_run_mcts_search_writes_artifacts(tmp_path, monkeypatch):
    import json as _json

    from skillopt.search import runner as runner_mod

    monkeypatch.setattr(runner_mod, "configure_models_from_cfg", lambda cfg, **k: ("x", "y", "z"))
    monkeypatch.setattr(runner_mod, "reset_token_tracker", lambda: None)
    monkeypatch.setattr(runner_mod, "get_token_summary", lambda: {"_total": {"total_tokens": 0}})
    monkeypatch.setattr(runner_mod, "SkillWorldModel", _RunnerMockWorldModel)

    out_root = str(tmp_path / "run")
    cfg = {
        "out_root": out_root,
        "seed": 42,
        "batch_size": 4,
        "edit_budget": 4,
        "merge_batch_size": 4,
        "analyst_workers": 4,
        "skill_update_mode": "patch",
        "skill_init": "",  # blank start
        "gate_metric": "soft",
        "search": {
            "algo": "mcts", "budget": 6, "branch": 2, "depth_limit": 2,
            "c_explore": 1.5, "lambda_cost": 0.0, "patience": 5, "max_iters": 20,
            "sel_eval_num": 6, "test_eval_num": 6, "eval_test": True, "efficiency_eps": 0.05,
        },
    }

    summary = runner_mod.run_mcts_search(cfg, _MockAdapter())

    # artifacts on disk
    assert (tmp_path / "run" / "mcts" / "tree.json").exists()
    assert (tmp_path / "run" / "mcts" / "frontier.json").exists()
    assert (tmp_path / "run" / "best_skill.md").exists()
    assert (tmp_path / "run" / "summary.json").exists()

    frontier = _json.loads((tmp_path / "run" / "mcts" / "frontier.json").read_text())["frontier"]
    assert len(frontier) >= 1
    # every frontier point carries selection + held-out test scores
    for pt in frontier:
        assert pt["test_success"] is not None
        assert pt["test_cost"] is not None
    assert summary["n_value_calls"] <= 6 + 2  # budget (soft cap + branch)
    assert summary["n_evaluated"] >= 1
    assert summary["value_mode"] == "paired"
    assert summary["n_task_rollouts"] >= 1


# ── CRN-paired Δ + successive halving (M2) ───────────────────────────────────

class _SHAdapter:
    """Deterministic per-task scores: harder items need more rules to solve.

    Item ``s-i`` has difficulty ``4 - (i % 5)`` (so the *first* tasks are the
    hardest → even a tiny prefix discriminates skills, the happy path for SH).
    A skill with ``n`` rules solves an item iff ``n >= difficulty``.  More rules
    ⇒ strictly higher success and higher cost — a clean signal for paired Δ / SH.
    """

    def rollout(self, items, skill, out_dir, **kw):
        n = skill.count("- rule")
        out = []
        for it in items:
            idx = int(str(it["id"]).rsplit("-", 1)[-1])
            hard = 1.0 if n >= (4 - (idx % 5)) else 0.0
            out.append({"id": it["id"], "hard": hard, "soft": hard, "cost_total_tokens": 50 + 20 * n})
        return out


def _sel_items(k=8):
    return [{"id": f"s-{i}"} for i in range(k)]


def _sk(skill, nid):
    return SkillNode(skill=skill, skill_hash=skill_hash(skill), node_id=nid)


def _rules(n):
    return "".join(f"\n- rule {i}" for i in range(n))


def test_crn_paired_delta_exact_at_full_coverage(tmp_path):
    from skillopt.search.value import PairedValueEstimator

    est = PairedValueEstimator(
        _SHAdapter(), _sel_items(8), out_root=str(tmp_path),
        gate_metric="hard", n0=4, sh_eta=2,
    )
    parent = _sk("", 0)
    child = _sk(_rules(2), 1)
    est.anchor(parent)
    # difficulties on s-0..s-7 = [4,3,2,1,0,4,3,2]; parent (0 rules) solves only
    # difficulty-0 items → s-4 → 1/8.
    assert parent.value.success == pytest.approx(0.125)

    est.estimate_children([child], parent, lambda_cost=0.0, cost_ref=parent.value.cost)
    # child (2 rules) solves difficulty <= 2 → s-2,s-3,s-4,s-7 → 4/8.
    # Telescoping must be exact (parent anchored on all of S):
    assert child.value.success == pytest.approx(0.5)
    assert child.value.cost == pytest.approx(90.0)          # 50 + 20*2
    # lone survivor was promoted to full coverage
    assert est.cache[child.skill_hash]["k"] == 8


def test_sh_prunes_weak_children_and_saves_rollouts(tmp_path):
    from skillopt.search.value import PairedValueEstimator

    est = PairedValueEstimator(
        _SHAdapter(), _sel_items(8), out_root=str(tmp_path),
        gate_metric="hard", n0=2, sh_eta=2,
    )
    parent = _sk("", 0)
    weak = _sk(_rules(1), 1)
    mid = _sk(_rules(2), 2)
    strong = _sk(_rules(5), 3)

    est.estimate_children([weak, mid, strong], parent, lambda_cost=0.0, cost_ref=50.0)

    k = lambda node: est.cache[node.skill_hash]["k"]
    # winner reaches full coverage; the two weak children are pruned before it
    assert k(strong) == 8
    assert k(weak) < 8 and k(mid) < 8
    assert min(k(weak), k(mid)) == 2          # one pruned at the first rung (n0=2)
    # every child still gets a telescoped value; the strongest scores highest
    assert all(c.value is not None for c in (weak, mid, strong))
    assert strong.value.success == max(weak.value.success, mid.value.success, strong.value.success)
    # SH saved rollouts vs full eval of every child on all 8 tasks (+8 anchor)
    assert est.n_task_rollouts < 8 + 3 * 8


def test_full_mode_evaluates_every_child_fully(tmp_path):
    from skillopt.search.value import PairedValueEstimator, ValueEstimator

    children_full = [_sk(_rules(1), 1), _sk(_rules(2), 2), _sk(_rules(5), 3)]
    children_paired = [_sk(_rules(1), 4), _sk(_rules(2), 5), _sk(_rules(5), 6)]
    parent = _sk("", 0)

    full = ValueEstimator(_SHAdapter(), _sel_items(8), out_root=str(tmp_path / "f"), gate_metric="hard")
    full.anchor(parent)
    full.estimate_children(children_full, parent)
    # full mode: every child evaluated on all 8 tasks
    assert all(full.cache[c.skill_hash]["k"] == 8 for c in children_full)
    assert full.n_task_rollouts == 8 + 3 * 8       # parent + 3 children × 8

    paired = PairedValueEstimator(
        _SHAdapter(), _sel_items(8), out_root=str(tmp_path / "p"),
        gate_metric="hard", n0=2, sh_eta=2,
    )
    paired.estimate_children(children_paired, parent=_sk("", 7), lambda_cost=0.0, cost_ref=50.0)
    assert paired.n_task_rollouts < full.n_task_rollouts   # paired is cheaper
