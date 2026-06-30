"""MCTS search entry point (mcts_03 §1, §7).

Orchestrates a cost-aware multi-objective MCTS run and writes the deliverables:

  * ``<out_root>/mcts/tree.json``      — every node + per-iteration log.
  * ``<out_root>/mcts/frontier.json``  — the success-cost Pareto menu, with
                                         held-out test scores per point.
  * ``<out_root>/best_skill.md``       — the ``perf`` prior pick (kept at the
                                         canonical path for eval_only/webui).
  * ``<out_root>/mcts/best_skill_efficiency.md`` — the ``efficiency`` prior pick.

Target + optimizer models stay frozen; deployment adds zero model calls.
"""
from __future__ import annotations

import json
import os
import time

from skillopt.engine.model_setup import configure_models_from_cfg
from skillopt.model import get_token_summary, reset_token_tracker
from skillopt.search.frontier import pareto_front, select_by_prior
from skillopt.search.mcts import MCTS
from skillopt.search.value import PairedValueEstimator, ValueEstimator
from skillopt.search.world_model import SkillWorldModel


def _load_initial_skill(cfg: dict) -> str:
    path = cfg.get("skill_init", "")
    if path and os.path.exists(os.path.abspath(path)):
        with open(os.path.abspath(path)) as f:
            content = f.read()
        print(f"  [initial skill] {path} ({len(content)} chars)")
        return content
    print("  [initial skill] no initial skill file — starting from blank")
    return ""


def run_mcts_search(cfg: dict, adapter) -> dict:
    out_root = cfg["out_root"]
    mcts_root = os.path.join(out_root, "mcts")
    os.makedirs(mcts_root, exist_ok=True)
    search = dict(cfg.get("search", {}))
    seed = int(cfg.get("seed", 42))

    print(f"\n{'='*60}")
    print("  MCTS-SkillOpt — cost-aware Pareto search")
    print(f"{'='*60}")
    print(
        f"  budget={search.get('budget')}  branch={search.get('branch')}  "
        f"depth={search.get('depth_limit')}  c={search.get('c_explore')}  "
        f"lambda={search.get('lambda_cost')}  sel_eval={search.get('sel_eval_num')}"
    )

    # ── one-time setup (shared with the linear trainer) ────────────────────
    t0 = time.time()
    configure_models_from_cfg(cfg)
    reset_token_tracker()
    adapter.setup(cfg)

    skill_init = _load_initial_skill(cfg)

    # ── fixed splits ───────────────────────────────────────────────────────
    sel_items = adapter.build_eval_env(
        env_num=int(search.get("sel_eval_num", 24)),
        split="valid_seen",
        seed=seed,
        out_root=out_root,
    )
    train_items = adapter.build_train_env(
        batch_size=int(cfg.get("batch_size", 40)),
        seed=seed,
        out_root=out_root,
    )
    print(f"  [splits] selection={len(sel_items)} train_batch={len(train_items)}")

    gate_metric = cfg.get("gate_metric", "hard")
    gate_mixed_weight = float(cfg.get("gate_mixed_weight", 0.5))

    value_mode = str(search.get("value_mode", "paired")).lower()
    if value_mode == "paired":
        value_estimator = PairedValueEstimator(
            adapter, sel_items,
            out_root=mcts_root,
            gate_metric=gate_metric,
            mixed_weight=gate_mixed_weight,
            n0=int(search.get("n0", 4)),
            sh_eta=int(search.get("sh_eta", 2)),
        )
    else:
        value_estimator = ValueEstimator(
            adapter, sel_items,
            out_root=mcts_root,
            gate_metric=gate_metric,
            mixed_weight=gate_mixed_weight,
        )
    print(f"  [value] mode={value_mode}  n0={search.get('n0')}  sh_eta={search.get('sh_eta')}")
    world_model = SkillWorldModel(
        adapter,
        update_mode=cfg.get("skill_update_mode", "patch"),
        edit_budget=int(cfg.get("edit_budget", 4)),
        merge_batch_size=int(cfg.get("merge_batch_size", 8)),
        analyst_workers=int(cfg.get("analyst_workers", 16)),
    )
    mcts = MCTS(
        world_model, value_estimator,
        out_root=mcts_root,
        train_items=train_items,
        branch=int(search.get("branch", 2)),
        depth_limit=int(search.get("depth_limit", 4)),
        c_explore=float(search.get("c_explore", 2.5)),
        lambda_cost=float(search.get("lambda_cost", 0.3)),
        budget=int(search.get("budget", 24)),
        patience=int(search.get("patience", 5)),
        max_iters=int(search.get("max_iters", 50)),
        expand_seed=int(search.get("expand_seed", 0)),
    )

    # ── search ─────────────────────────────────────────────────────────────
    root, evaluated, front = mcts.run(skill_init)
    print(
        f"\n  [search done] nodes={len(mcts.nodes)} evaluated={len(evaluated)} "
        f"candidates={value_estimator.n_evals} task-rollouts={value_estimator.n_task_rollouts} "
        f"frontier={len(front)}"
    )

    # ── held-out test eval of frontier points (mcts_04 §6: top-k full) ─────
    test_scores: dict[str, dict] = {}
    if search.get("eval_test", True):
        test_items = adapter.build_eval_env(
            env_num=int(search.get("test_eval_num", 0)),
            split="valid_unseen",
            seed=seed,
            out_root=out_root,
        )
        test_estimator = ValueEstimator(
            adapter, test_items,
            out_root=os.path.join(mcts_root, "test_eval"),
            gate_metric=gate_metric,
            mixed_weight=gate_mixed_weight,
        )
        print(f"  [test] evaluating {len(front)} frontier points on {len(test_items)} items")
        for node in front:
            # Non-mutating: keeps node.value as the *selection* value so the
            # frontier (extracted on selection scores) stays split-correct.
            rec = test_estimator.eval_content(node.skill_hash, node.skill)
            test_scores[node.skill_hash] = {"success": rec["success"], "cost": rec["cost"]}

    # ── artifacts ──────────────────────────────────────────────────────────
    artifacts = _write_artifacts(
        out_root, mcts_root, mcts, root, evaluated, front, test_scores, search,
    )

    # Full per-stage token breakdown so the cost profile (target rollout vs
    # optimizer) is reproducible without re-deriving it from disk.
    token_stages = get_token_summary()
    total_tokens = token_stages.get("_total", {}).get("total_tokens", 0)
    target_stage = token_stages.get("rollout", {})
    target_tokens = target_stage.get("total_tokens", 0)
    summary = {
        "algo": "mcts",
        "value_mode": value_mode,
        "n_nodes": len(mcts.nodes),
        "n_evaluated": len(evaluated),
        "n_iterations": len(mcts.iter_log),
        "n_value_calls": value_estimator.n_evals,
        "n_task_rollouts": value_estimator.n_task_rollouts,
        "cost_ref": mcts.cost_ref,
        "frontier": artifacts["frontier"],
        "perf_skill": artifacts["perf_path"],
        "efficiency_skill": artifacts["efficiency_path"],
        "wall_time_s": round(time.time() - t0, 1),
        "tokens_total": total_tokens,
        "tokens_target_rollout": target_tokens,
        "tokens_optimizer": max(0, total_tokens - target_tokens),
        "tokens_by_stage": {
            stage: vals.get("total_tokens", 0)
            for stage, vals in token_stages.items()
            if stage != "_total"
        },
    }
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Output saved to: {out_root}")
    _print_frontier(artifacts["frontier"])
    return summary


def _write_artifacts(out_root, mcts_root, mcts, root, evaluated, front, test_scores, search):
    # tree.json
    tree = {
        "cost_ref": mcts.cost_ref,
        "lambda_cost": mcts.lambda_cost,
        "root_id": root.node_id,
        "nodes": [n.to_record() for n in mcts.nodes],
        "iter_log": mcts.iter_log,
    }
    with open(os.path.join(mcts_root, "tree.json"), "w") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)

    # frontier.json
    frontier_records = []
    for node in front:
        node_dir = os.path.join(mcts_root, "nodes", f"{node.node_id:04d}")
        skill_path = os.path.join(node_dir, "skill.md")
        if not os.path.exists(skill_path):
            os.makedirs(node_dir, exist_ok=True)
            with open(skill_path, "w") as f:
                f.write(node.skill)
        ts = test_scores.get(node.skill_hash, {})
        frontier_records.append({
            "node_id": node.node_id,
            "depth": node.depth,
            "skill_hash": node.skill_hash,
            "skill_len": len(node.skill),
            "skill_path": skill_path,
            "sel_success": node.value.success,
            "sel_cost": node.value.cost,
            "test_success": ts.get("success"),
            "test_cost": ts.get("cost"),
        })
    with open(os.path.join(mcts_root, "frontier.json"), "w") as f:
        json.dump({"frontier": frontier_records}, f, indent=2, ensure_ascii=False)

    # perf / efficiency best_skill.md
    eps = float(search.get("efficiency_eps", 0.01))
    perf_node = select_by_prior(front, "perf", eps)
    eff_node = select_by_prior(front, "efficiency", eps)

    perf_path = os.path.join(out_root, "best_skill.md")  # canonical path
    if perf_node is not None:
        with open(perf_path, "w") as f:
            f.write(perf_node.skill)
    eff_path = os.path.join(mcts_root, "best_skill_efficiency.md")
    if eff_node is not None:
        with open(eff_path, "w") as f:
            f.write(eff_node.skill)

    return {
        "frontier": frontier_records,
        "perf_path": perf_path,
        "efficiency_path": eff_path,
        "perf_node": None if perf_node is None else perf_node.node_id,
        "efficiency_node": None if eff_node is None else eff_node.node_id,
    }


def _print_frontier(frontier_records: list[dict]) -> None:
    print("\n  success-cost Pareto frontier (selection / test):")
    print(f"  {'node':>5} {'depth':>5} {'sel_succ':>9} {'sel_cost':>10} {'test_succ':>10} {'test_cost':>10}")
    for r in frontier_records:
        ts = "-" if r["test_success"] is None else f"{r['test_success']:.4f}"
        tc = "-" if r["test_cost"] is None else f"{r['test_cost']:.1f}"
        print(
            f"  {r['node_id']:>5} {r['depth']:>5} {r['sel_success']:>9.4f} "
            f"{r['sel_cost']:>10.1f} {ts:>10} {tc:>10}"
        )
