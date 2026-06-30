# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

SkillOpt trains an agent's **skill document** (a Markdown file) like a neural net — with epochs, batch size, learning rate, and a validation gate — while the target model stays frozen. An optimizer model turns scored rollouts into bounded add/delete/replace edits; an edit is accepted only if it strictly improves a held-out validation score. The deployed artifact is a compact `best_skill.md` that adds zero inference-time model calls.

Two **independent** top-level packages live here:
- `skillopt/` — the research/paper code (the full training loop and benchmarks).
- `skillopt_sleep/` — a decoupled open-source deployment-time tool (nightly "sleep cycle" for local coding agents). It has **zero dependency** on `skillopt/`; the validation gate is vendored. Do not introduce imports between the two.
- `skillopt_webui/` — Gradio dashboard.

Note: the training loop is internally named **ReflACT** in many docstrings/identifiers — it is the same thing as SkillOpt's loop, not a separate system.

## Common commands

```bash
pip install -e ".[dev]"          # base + ruff/pytest
pip install -e ".[alfworld]"     # benchmark-specific extras (also: claude, qwen, webui, docs, all)

# Train (any YAML key is overridable as a --flag)
python scripts/train.py --config configs/searchqa/default.yaml \
    --optimizer_model gpt-5.5 --target_model gpt-5.5 --out_root outputs/run1
# Convenience launchers (set OPTIMIZER_MODEL/TARGET_MODEL env, pass extra flags through):
bash scripts/run_searchqa.sh --num_epochs 2

# Eval a single skill, no training
python scripts/eval_only.py --config configs/spreadsheetbench/default.yaml \
    --skill skillopt/envs/spreadsheetbench/skills/initial.md --out_root outputs/eval1

# Console entry points (installed): skillopt-train, skillopt-eval

# Tests / lint
pytest                  # run all
pytest tests/test_scoring.py::test_name   # single test
ruff check .            # line-length 120, E501 ignored

# Sleep tool deterministic smoke test (no API key needed)
python -m skillopt_sleep.experiments.run_experiment --persona researcher --assert-improves

# Docs site
mkdocs serve            # needs ".[docs]"
```

A fast iteration recipe for the training loop (tiny everything) is documented in the `scripts/train.py` module docstring.

## Configuration system

- Configs are YAML under `configs/<env>/default.yaml`, inheriting from `configs/_base_/default.yaml` via the `_base_:` key.
- **Every** YAML key can be overridden on the CLI with `--key value` (e.g. `--batch_size 40 --num_epochs 2 --learning_rate 6`). This is the primary knob surface — read `configs/_base_/default.yaml` for the full set; sections are `model`, `train`, `gradient`, `optimizer`, `evaluation`, `env`.
- `optimizer.learning_rate` = max edits per training step ("edit budget"). `lr_scheduler` (constant/linear/cosine/autonomous) and `skill_update_mode` (patch / rewrite_from_suggestions / full_rewrite_minibatch) control how edits are produced.
- Credentials come from env vars / `.env` (Azure OpenAI, MiniMax, etc.); the matching `model.*_api_key`/`*_endpoint` config fields are fallbacks. Never commit keys.

## Architecture

**Six-stage per-step loop** (`skillopt/engine/trainer.py`, environment-agnostic):
`rollout → reflect → aggregate → select → update → evaluate(gate)`. The trainer owns scheduling, the slow update (momentum), meta-skill (optimizer memory), and the validation gate; it delegates all env-specific work to an `EnvAdapter`.

**Pipeline modules** (the "stages"):
- `skillopt/gradient/` — `reflect.py` (analyze trajectories → patches), `aggregate.py` (hierarchical merge).
- `skillopt/optimizer/` — `clip.py` (rank/select edits), `skill.py` (apply patch), `rewrite.py`, `scheduler.py` (LR schedules), `slow_update.py`, `meta_skill.py`, `skill_aware.py` (skill-defect vs execution-lapse split with a protected appendix).
- `skillopt/evaluation/gate.py` — the accept/reject validation gate.

**Environments** (`skillopt/envs/<name>/`): each benchmark is a self-contained package with `adapter.py` (subclass of `skillopt/envs/base.py:EnvAdapter`), `rollout.py`, `reflect.py`, `evaluator.py`, `dataloader.py`, plus `skills/initial.md` (seed skill) and `prompts/`. Adapters are registered lazily in `scripts/train.py:_register_builtins()` so missing optional deps just skip that env. To add a benchmark, copy `skillopt/envs/_template/` (see `docs/guide/new-benchmark.md`).

**Model backends** (`skillopt/model/`): `router.py` selects the active backend; modules include `azure_openai.py`, `codex_backend.py`/`codex_harness.py`, `claude_backend.py`, `qwen_backend.py`, `minimax_backend.py`. Backends are chosen per-role via `model.optimizer_backend` / `model.target_backend` in config. Add one per `docs/guide/new-backend.md`.

**Prompts** are Markdown files in `skillopt/prompts/` (and per-env `prompts/`), loaded via `load_prompt`. Edit prompt behavior there rather than hardcoding strings.

**Data & outputs**: dataset splits live in `data/<bench>_*_split/`; training writes run dirs to `outputs/`; released/checkpoint skills are under `ckpt/`. The split format and item JSON schema are documented in `docs/guideline.html` (§4).

## Conventions

- Use type hints on function signatures; keep docstrings concise; follow existing patterns in the touched module.
- Test changes against an existing benchmark before submitting (CONTRIBUTING.md).
- Keep `skillopt/` and `skillopt_sleep/` decoupled.
