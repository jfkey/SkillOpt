"""Skill world model: state transition = one SkillOpt edit step (mcts_03 §2).

``transition`` reassembles SkillOpt's stages ②③④⑤ (reflect → aggregate →
select → apply) on top of the *same* importable stage functions the linear
trainer uses — it does **not** fork ``trainer.py``.  Stage ① (the train
rollout that feeds reflect) is provided once per node via
``rollout_train_evidence`` and cached on the node, so a node expanded into ``b``
children pays for its train rollout only once.

The gate is intentionally absent here — accept/reject is the search's job, not
the transition's (the linear trainer keeps its own gate).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from skillopt.gradient.aggregate import merge_patches
from skillopt.optimizer.clip import rank_and_select
from skillopt.optimizer.skill import apply_patch_with_report
from skillopt.optimizer.update_modes import get_payload_items, normalize_update_mode


@dataclass
class TransitionResult:
    candidate_skill: str
    ranked_patch: dict
    apply_report: list


def normalise_patches(
    raw_patches: list,
    update_mode: str = "patch",
) -> tuple[list[dict], list[dict]]:
    """Split raw analyst patches into failure / success payload lists.

    Faithful to ``trainer._normalise_patches``: pulls the inner ``patch``
    sub-dict, drops empty payloads, stamps ``source_type`` / ``support_count``
    onto each item, and routes by ``source_type``.
    """
    mode = normalize_update_mode(update_mode)
    failure: list[dict] = []
    success: list[dict] = []
    for p in raw_patches:
        if not isinstance(p, dict):
            continue
        inner = p.get("patch", p)
        if not isinstance(inner, dict):
            continue
        items = get_payload_items(inner, mode)
        if not items:
            continue
        support = max(int(p.get("batch_size", 0) or 0), 1)
        for item in items:
            if isinstance(item, dict):
                item.setdefault("source_type", p.get("source_type", "failure"))
                item.setdefault("support_count", support)
        if p.get("source_type", "failure") == "success":
            success.append(inner)
        else:
            failure.append(inner)
    return failure, success


class SkillWorldModel:
    """Wraps ``transition`` (edit step) over an EnvAdapter."""

    def __init__(
        self,
        adapter,
        *,
        update_mode: str = "patch",
        edit_budget: int = 4,
        merge_batch_size: int = 8,
        analyst_workers: int = 16,
    ) -> None:
        self.adapter = adapter
        self.update_mode = normalize_update_mode(update_mode)
        self.edit_budget = edit_budget
        self.merge_batch_size = merge_batch_size
        self.analyst_workers = analyst_workers

    def rollout_train_evidence(self, skill: str, train_items: list, out_dir: str) -> list:
        """Stage ①: roll the skill on a train batch to produce reflect evidence."""
        return self.adapter.rollout(train_items, skill, out_dir)

    def transition(
        self,
        parent_skill: str,
        train_results: list,
        *,
        train_rollout_dir: str,
        expand_dir: str,
        seed: int,
    ) -> TransitionResult | None:
        """Stages ②③④⑤: produce one candidate child skill from a parent.

        Returns ``None`` when reflect yields no usable edits (a no-op step).
        """
        os.makedirs(expand_dir, exist_ok=True)
        pred_dir = os.path.join(train_rollout_dir, "predictions")
        patches_dir = os.path.join(expand_dir, "patches")

        # ② REFLECT
        raw_patches = self.adapter.reflect(
            train_results,
            parent_skill,
            expand_dir,
            prediction_dir=pred_dir,
            patches_dir=patches_dir,
            random_seed=seed,
            step_buffer_context="",
            meta_skill_context="",
        )
        failure, success = normalise_patches(raw_patches, self.update_mode)
        if not failure and not success:
            return None

        # ③ AGGREGATE
        merged_patch = merge_patches(
            parent_skill,
            failure,
            success,
            batch_size=self.merge_batch_size,
            verbose=False,
            workers=self.analyst_workers,
            update_mode=self.update_mode,
        )

        # ④ SELECT (clip to edit budget)
        ranked_patch = rank_and_select(
            parent_skill,
            merged_patch,
            max_edits=self.edit_budget,
            update_mode=self.update_mode,
        )

        # ⑤ APPLY
        candidate_skill, apply_report = apply_patch_with_report(parent_skill, ranked_patch)
        return TransitionResult(candidate_skill, ranked_patch, apply_report)
