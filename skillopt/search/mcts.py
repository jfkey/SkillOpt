"""UCT search over the skill-version tree (mcts_01, mcts_05 Algorithm 1).

Four phases per iteration:
  1. SELECTION  — UCT descent from root (pure bookkeeping, 0 model calls).
  2. EXPANSION  — grow ``b`` candidate children = ``b`` edit steps.
  3. SIMULATION — estimate each child's 2-D value (the only expensive step).
  4. BACKPROP   — accumulate the 2-D value vector ``W`` up the path (Option B:
                  λ is *not* baked in; ``Q`` is scalarized on demand at select).

Frontier extraction is λ-free: the non-dominated set over all evaluated nodes'
absolute 2-D values.  Navigation (mean-value ``Q``) and extraction (global 2-D
ledger) are deliberately separated (mcts_01 §5).
"""
from __future__ import annotations

import math
import os

from skillopt.search.frontier import pareto_front
from skillopt.search.node import NodeValue, SkillNode, scalarize
from skillopt.search.value import ValueEstimator
from skillopt.search.world_model import SkillWorldModel
from skillopt.utils import skill_hash


class MCTS:
    def __init__(
        self,
        world_model: SkillWorldModel,
        value_estimator: ValueEstimator,
        *,
        out_root: str,
        train_items: list,
        branch: int = 2,
        depth_limit: int = 4,
        c_explore: float = 2.5,
        lambda_cost: float = 0.3,
        budget: int = 24,
        patience: int = 5,
        max_iters: int = 50,
        expand_seed: int = 0,
    ) -> None:
        self.world_model = world_model
        self.value_estimator = value_estimator
        self.out_root = out_root
        self.train_items = train_items
        self.branch = branch
        self.depth_limit = depth_limit
        self.c_explore = c_explore
        self.lambda_cost = lambda_cost
        self.budget = budget
        self.patience = patience
        self.max_iters = max_iters
        self.expand_seed = expand_seed

        self.cost_ref: float = 1.0  # no-skill baseline; set after root eval
        self.nodes: list[SkillNode] = []
        self.iter_log: list[dict] = []
        self._next_id = 0

    # ── node factory ──────────────────────────────────────────────────────
    def _new_node(self, skill, parent=None, depth=0, edge_edits=None) -> SkillNode:
        node = SkillNode(
            skill=skill,
            skill_hash=skill_hash(skill),
            depth=depth,
            node_id=self._next_id,
            parent=parent,
            edge_edits=edge_edits,
        )
        self._next_id += 1
        self.nodes.append(node)
        return node

    def _node_dir(self, node: SkillNode) -> str:
        return os.path.join(self.out_root, "nodes", f"{node.node_id:04d}")

    # ── 1. SELECTION ──────────────────────────────────────────────────────
    def uct(self, ch: SkillNode) -> float:
        if ch.terminal:
            return float("-inf")
        if ch.N == 0:
            return float("inf")  # force at least one visit
        mean = NodeValue(ch.W_success / ch.N, ch.W_cost / ch.N)
        q = scalarize(mean, self.lambda_cost, self.cost_ref)
        parent_n = ch.parent.N if ch.parent and ch.parent.N > 0 else 1
        explore = self.c_explore * math.sqrt(math.log(parent_n) / ch.N)
        return q + explore

    def select(self, root: SkillNode) -> SkillNode:
        node = root
        while node.expanded and node.children and node.depth < self.depth_limit:
            best = max(node.children, key=self.uct)
            if self.uct(best) == float("-inf"):
                break  # every child terminal — stop here
            node = best
        return node

    # ── 2. EXPANSION ──────────────────────────────────────────────────────
    def expand(self, node: SkillNode, seen: set[str]) -> list[SkillNode]:
        if node.train_evidence is None:
            tr_dir = os.path.join(self._node_dir(node), "train_rollout")
            node.train_rollout_dir = tr_dir
            node.train_evidence = self.world_model.rollout_train_evidence(
                node.skill, self.train_items, tr_dir,
            )
        children: list[SkillNode] = []
        for k in range(self.branch):
            expand_dir = os.path.join(self._node_dir(node), f"expand_{k}")
            tr = self.world_model.transition(
                node.skill,
                node.train_evidence,
                train_rollout_dir=node.train_rollout_dir,
                expand_dir=expand_dir,
                seed=self.expand_seed + k,
            )
            if tr is None:
                continue
            h = skill_hash(tr.candidate_skill)
            if h == node.skill_hash or h in seen:
                # no-op edit (identical to parent) or transposition — skip,
                # don't spend a value-estimation call on it.
                continue
            seen.add(h)
            child = self._new_node(
                tr.candidate_skill, parent=node, depth=node.depth + 1,
                edge_edits=tr.ranked_patch,
            )
            node.children.append(child)  # attach so SELECT can descend
            children.append(child)
        node.expanded = True
        return children

    # ── 4. BACKPROP (Option B: 2-D vector, λ not baked in) ────────────────
    def backup(self, child: SkillNode) -> None:
        x: SkillNode | None = child
        while x is not None:
            x.N += 1
            x.W_success += child.value.success
            x.W_cost += child.value.cost
            x = x.parent

    # ── main loop ─────────────────────────────────────────────────────────
    def run(self, root_skill: str):
        root = self._new_node(root_skill, depth=0)
        self.value_estimator.anchor(root)  # full absolute anchor (mcts_01 §5)
        self.cost_ref = root.value.cost if root.value.cost > 0 else 1.0
        seen: set[str] = {root.skill_hash}
        evaluated: list[SkillNode] = [root]
        no_improve = 0
        iters = 0

        while (
            self.value_estimator.n_evals < self.budget
            and no_improve < self.patience
            and iters < self.max_iters
        ):
            iters += 1
            node = self.select(root)

            # No expandable frontier under this node → it is a dead leaf.
            if node.expanded or node.depth >= self.depth_limit:
                node.terminal = True
                if node is root:
                    break
                continue

            children = self.expand(node, seen)
            if not children:
                node.terminal = True
                continue

            # 3. SIMULATION — value the b siblings together so the estimator can
            #    share the parent CRN anchor and prune across them (SH).  In
            #    full mode this is just b independent absolute evals.
            self.value_estimator.estimate_children(
                children, node, lambda_cost=self.lambda_cost, cost_ref=self.cost_ref,
            )
            # 4. BACKPROP — accumulate each child's 2-D value up its path.
            for child in children:
                evaluated.append(child)
                self.backup(child)
                self._write_skill(child)

            front_after = pareto_front(evaluated)
            front_after_hashes = {n.skill_hash for n in front_after}
            grew = any(c.skill_hash in front_after_hashes for c in children)
            no_improve = 0 if grew else no_improve + 1

            self.iter_log.append({
                "iter": iters,
                "expanded_node": node.node_id,
                "depth": node.depth,
                "n_children": len(children),
                "n_evals": self.value_estimator.n_evals,
                "frontier_size": len(front_after),
                "frontier_grew": grew,
                "no_improve": no_improve,
            })

        return root, evaluated, pareto_front(evaluated)

    def _write_skill(self, node: SkillNode) -> None:
        node_dir = self._node_dir(node)
        os.makedirs(node_dir, exist_ok=True)
        with open(os.path.join(node_dir, "skill.md"), "w") as f:
            f.write(node.skill)
