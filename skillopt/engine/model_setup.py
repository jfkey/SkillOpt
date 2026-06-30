"""Shared model-backend configuration.

Both the linear :class:`~skillopt.engine.trainer.ReflACTTrainer` and the MCTS
search runner (:mod:`skillopt.search`) must configure the optimizer/target
model backends identically before any rollout/reflect call.  This module owns
that one-time setup so the two entry points cannot drift apart.

The function reads a *flat* config dict (post ``flatten_config``), mutates the
global model-router state (deployments, backends, reasoning effort), and back-
fills ``cfg["optimizer_backend"]`` / ``cfg["target_backend"]`` when they were
left unset.  It is intentionally side-effecting and returns the resolved
backend triple for logging.
"""
from __future__ import annotations

import os

from skillopt.model import (
    configure_azure_openai,
    configure_claude_code_exec,
    configure_codex_exec,
    configure_minimax_chat,
    configure_qwen_chat,
    set_reasoning_effort,
    set_optimizer_backend,
    set_optimizer_deployment,
    set_target_backend,
    set_target_deployment,
)


def configure_models_from_cfg(cfg: dict, *, verbose: bool = True) -> tuple[str, str, str]:
    """Configure optimizer/target backends from a flat config dict.

    Returns ``(backend, optimizer_backend, target_backend)``.  Mutates
    ``cfg`` in place to record the resolved ``optimizer_backend`` /
    ``target_backend`` and the global model-router state.
    """
    backend = cfg.get("model_backend", "azure_openai")
    configure_azure_openai(
        endpoint=(
            cfg.get("azure_openai_endpoint")
            or cfg.get("azure_endpoint")
            or None
        ),
        api_version=(
            cfg.get("azure_openai_api_version")
            or cfg.get("azure_api_version")
            or None
        ),
        api_key=(
            cfg.get("azure_openai_api_key")
            or cfg.get("azure_api_key")
            or None
        ),
        auth_mode=cfg.get("azure_openai_auth_mode") or None,
        ad_scope=cfg.get("azure_openai_ad_scope") or None,
        managed_identity_client_id=cfg.get("azure_openai_managed_identity_client_id") or None,
        optimizer_endpoint=cfg.get("optimizer_azure_openai_endpoint") or None,
        optimizer_api_version=cfg.get("optimizer_azure_openai_api_version") or None,
        optimizer_api_key=cfg.get("optimizer_azure_openai_api_key") or None,
        optimizer_auth_mode=cfg.get("optimizer_azure_openai_auth_mode") or None,
        optimizer_ad_scope=cfg.get("optimizer_azure_openai_ad_scope") or None,
        optimizer_managed_identity_client_id=(
            cfg.get("optimizer_azure_openai_managed_identity_client_id") or None
        ),
        target_endpoint=cfg.get("target_azure_openai_endpoint") or None,
        target_api_version=cfg.get("target_azure_openai_api_version") or None,
        target_api_key=cfg.get("target_azure_openai_api_key") or None,
        target_auth_mode=cfg.get("target_azure_openai_auth_mode") or None,
        target_ad_scope=cfg.get("target_azure_openai_ad_scope") or None,
        target_managed_identity_client_id=(
            cfg.get("target_azure_openai_managed_identity_client_id") or None
        ),
    )
    optimizer_backend = cfg.get("optimizer_backend")
    target_backend = cfg.get("target_backend")
    if not optimizer_backend or not target_backend:
        if backend in {"claude", "claude_chat"}:
            optimizer_backend = optimizer_backend or "claude_chat"
            target_backend = target_backend or "claude_chat"
        elif backend in {"codex", "codex_exec"}:
            optimizer_backend = optimizer_backend or "openai_chat"
            target_backend = target_backend or "codex_exec"
        elif backend == "claude_code_exec":
            optimizer_backend = optimizer_backend or "openai_chat"
            target_backend = target_backend or "claude_code_exec"
        elif backend in {"qwen", "qwen_chat"}:
            optimizer_backend = optimizer_backend or "openai_chat"
            target_backend = target_backend or "qwen_chat"
        else:
            optimizer_backend = optimizer_backend or "openai_chat"
            target_backend = target_backend or "openai_chat"
        cfg["optimizer_backend"] = optimizer_backend
        cfg["target_backend"] = target_backend
    set_optimizer_backend(optimizer_backend)
    set_target_backend(target_backend)
    set_optimizer_deployment(cfg["optimizer_model"])
    set_target_deployment(cfg["target_model"])
    configure_codex_exec(
        path=cfg.get("codex_exec_path", "codex"),
        sandbox=cfg.get("codex_exec_sandbox", "workspace-write"),
        profile=cfg.get("codex_exec_profile", ""),
        full_auto=cfg.get("codex_exec_full_auto", False),
        reasoning_effort=cfg.get("codex_exec_reasoning_effort", "none"),
        use_sdk=cfg.get("codex_exec_use_sdk", None),
        network_access=cfg.get("codex_exec_network_access", False),
        web_search=cfg.get("codex_exec_web_search", False),
        approval_policy=cfg.get("codex_exec_approval_policy", "never"),
    )
    configure_claude_code_exec(
        path=cfg.get("claude_code_exec_path", "claude"),
        profile=cfg.get("claude_code_exec_profile", ""),
        use_sdk=cfg.get("claude_code_exec_use_sdk", None),
        effort=cfg.get("claude_code_exec_effort", cfg.get("reasoning_effort", "medium")),
        max_thinking_tokens=cfg.get("claude_code_exec_max_thinking_tokens", 16384),
    )
    configure_qwen_chat(
        base_url=cfg.get("qwen_chat_base_url") or None,
        api_key=cfg.get("qwen_chat_api_key") or None,
        temperature=cfg.get("qwen_chat_temperature"),
        timeout_seconds=cfg.get("qwen_chat_timeout_seconds"),
        max_tokens=cfg.get("qwen_chat_max_tokens"),
        enable_thinking=cfg.get("qwen_chat_enable_thinking"),
        optimizer_base_url=cfg.get("optimizer_qwen_chat_base_url") or None,
        optimizer_api_key=cfg.get("optimizer_qwen_chat_api_key") or None,
        optimizer_temperature=cfg.get("optimizer_qwen_chat_temperature"),
        optimizer_timeout_seconds=cfg.get("optimizer_qwen_chat_timeout_seconds"),
        optimizer_max_tokens=cfg.get("optimizer_qwen_chat_max_tokens"),
        optimizer_enable_thinking=cfg.get("optimizer_qwen_chat_enable_thinking"),
        target_base_url=cfg.get("target_qwen_chat_base_url") or None,
        target_api_key=cfg.get("target_qwen_chat_api_key") or None,
        target_temperature=cfg.get("target_qwen_chat_temperature"),
        target_timeout_seconds=cfg.get("target_qwen_chat_timeout_seconds"),
        target_max_tokens=cfg.get("target_qwen_chat_max_tokens"),
        target_enable_thinking=cfg.get("target_qwen_chat_enable_thinking"),
    )
    configure_minimax_chat(
        base_url=cfg.get("minimax_base_url") or None,
        api_key=cfg.get("minimax_api_key") or None,
        temperature=cfg.get("minimax_temperature"),
        max_tokens=cfg.get("minimax_max_tokens"),
        enable_thinking=cfg.get("minimax_enable_thinking"),
    )
    minimax_model_cfg = cfg.get("minimax_model")
    if minimax_model_cfg and cfg.get("target_backend") == "minimax_chat":
        set_target_deployment(str(minimax_model_cfg))
    os.environ["REFLACT_CODEX_TRACE_TO_OPTIMIZER"] = (
        "1"
        if target_backend == "codex_exec" and cfg.get("codex_trace_to_optimizer", False)
        else "0"
    )
    reasoning = cfg.get("reasoning_effort", "") or None
    set_reasoning_effort(reasoning)
    if verbose:
        print(
            f"  [model config] backend={backend}  "
            f"optimizer={cfg['optimizer_model']} ({optimizer_backend})  "
            f"target={cfg['target_model']} ({target_backend})  "
            f"reasoning={reasoning or 'off'}"
        )
    return backend, optimizer_backend, target_backend
