#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# SkillOpt — SearchQA training launch script (gpt-4.1 variant)
#
# Same full-size config as run_searchqa.sh, but targets gpt-4.1 for both the
# optimizer and target roles. gpt-4.1 does not support reasoning_effort, so it
# is forced to empty to avoid the Azure 400 error.
#
# Usage:
#   bash scripts/run_searchqa_gpt41.sh
#   bash scripts/run_searchqa_gpt41.sh --num_epochs 2 --edit_budget 6
#   bash scripts/run_searchqa_gpt41.sh --split_dir /path/to/searchqa_split
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-gpt-4.1}"
TARGET_MODEL="${TARGET_MODEL:-gpt-4.1}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEFAULT_OUT_ROOT="${PROJECT_ROOT}/outputs/skillopt_searchqa_${TARGET_MODEL}_${TIMESTAMP}"

echo "============================================================"
echo "  SkillOpt — SearchQA Training (gpt-4.1)"
echo "============================================================"
echo "  Optimizer:  ${OPTIMIZER_MODEL}"
echo "  Target:  ${TARGET_MODEL}"
echo "============================================================"

cd "${PROJECT_ROOT}"

python scripts/train.py \
    --config configs/searchqa/default.yaml \
    --optimizer_model "${OPTIMIZER_MODEL}" \
    --target_model "${TARGET_MODEL}" \
    --reasoning_effort "" \
    --out_root "${DEFAULT_OUT_ROOT}" \
    "$@"

echo ""
echo "Done! Results saved to: ${DEFAULT_OUT_ROOT}"
