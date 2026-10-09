"""Configuration and credential management for the 4-API-Key Consensus Pipeline.

Supports four independent AI API keys for multi-stage consensus:
  - Stage 1 (Discovery): AI_API_KEY_1 (or GEMINI_API_KEY_1)
  - Stage 2 (Independent Evaluation): AI_API_KEY_2 (or GEMINI_API_KEY_2)
  - Stage 3 (Independent Evaluation): AI_API_KEY_3 (or GEMINI_API_KEY_3)
  - Stage 4 (Final Consensus Review): AI_API_KEY_4 (or GEMINI_API_KEY_4)

Never logs, prints, or exposes raw API keys.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

# Default model used across consensus stages if not overridden per-stage
DEFAULT_CONSENSUS_MODEL = os.environ.get("GEMINI_MODEL") or "gemini-3.8-flash"
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_SEMANTIC_THRESHOLD = 0.35
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 3

# Configurable ranking weights
DEFAULT_WEIGHT_AGREEMENT = 0.35
DEFAULT_WEIGHT_DISCOVERY = 0.25
DEFAULT_WEIGHT_REVIEWER = 0.25
DEFAULT_WEIGHT_HOOK = 0.15
DEFAULT_REDUNDANCY_PENALTY = 15.0


def mask_key(key: Optional[str]) -> str:
    """Mask an API key for safe debugging/status messages. Never reveals the secret."""
    if not key:
        return "<unset>"
    s = str(key).strip()
    if len(s) <= 8:
        return "***"
    return f"{s[:3]}...{s[-4:]}"


def get_consensus_api_key(stage: int) -> Optional[str]:
    """Retrieve the API key for a given stage (1 to 4).

    Order of precedence:
      1. AI_API_KEY_{stage}
      2. GEMINI_API_KEY_{stage}
      3. If stage == 1 and single-key fallback allowed, GEMINI_API_KEY
    """
    if stage not in (1, 2, 3, 4):
        raise ValueError(f"Invalid consensus stage: {stage}. Must be 1, 2, 3, or 4.")

    val = os.environ.get(f"AI_API_KEY_{stage}")
    if val and val.strip():
        return val.strip()

    val = os.environ.get(f"GEMINI_API_KEY_{stage}")
    if val and val.strip():
        return val.strip()

    return None


def get_consensus_model(stage: int) -> str:
    """Retrieve the model name for a given stage (1 to 4)."""
    if stage not in (1, 2, 3, 4):
        raise ValueError(f"Invalid consensus stage: {stage}. Must be 1, 2, 3, or 4.")

    stage_model = os.environ.get(f"AI_MODEL_{stage}") or os.environ.get(f"GEMINI_MODEL_{stage}")
    if stage_model and stage_model.strip():
        return stage_model.strip()

    return os.environ.get("GEMINI_MODEL") or DEFAULT_CONSENSUS_MODEL


def is_consensus_enabled() -> bool:
    """Check if the 4-key consensus pipeline is active.

    Active if:
      - Explicitly enabled via CONSENSUS_PIPELINE=1 / AI_CONSENSUS=1
      - OR all four independent keys are present in the environment.
    """
    explicit = os.environ.get("CONSENSUS_PIPELINE") or os.environ.get("AI_CONSENSUS")
    if explicit is not None:
        return explicit.strip().lower() in ("1", "true", "yes", "on", "enabled")

    # Auto-enable if all 4 keys exist
    has_all_4 = all(get_consensus_api_key(s) is not None for s in (1, 2, 3, 4))
    return has_all_4


def is_strict_consensus() -> bool:
    """Check if strict 3-stage consensus is required (default: True).

    When True, only clips independently discovered by ALL THREE stages (1, 2, 3)
    are eligible for final consensus approval by Stage 4.
    """
    val = os.environ.get("CONSENSUS_STRICT", "1").strip().lower()
    return val not in ("0", "false", "no", "off", "disabled")


def get_iou_threshold() -> float:
    """Intersection-over-Union threshold for temporal overlap matching."""
    try:
        val = float(os.environ.get("CONSENSUS_IOU_THRESHOLD", DEFAULT_IOU_THRESHOLD))
        return max(0.1, min(0.95, val))
    except (ValueError, TypeError):
        return DEFAULT_IOU_THRESHOLD


def get_semantic_threshold() -> float:
    """Semantic similarity threshold for secondary matching."""
    try:
        val = float(os.environ.get("CONSENSUS_SEMANTIC_THRESHOLD", DEFAULT_SEMANTIC_THRESHOLD))
        return max(0.05, min(0.95, val))
    except (ValueError, TypeError):
        return DEFAULT_SEMANTIC_THRESHOLD


def get_timeout_seconds() -> float:
    """Timeout in seconds for consensus API calls."""
    try:
        return float(os.environ.get("CONSENSUS_TIMEOUT", DEFAULT_TIMEOUT_SECONDS))
    except (ValueError, TypeError):
        return DEFAULT_TIMEOUT_SECONDS


def get_max_retries() -> int:
    """Maximum bounded retries with exponential backoff."""
    try:
        return max(1, int(os.environ.get("CONSENSUS_MAX_RETRIES", DEFAULT_MAX_RETRIES)))
    except (ValueError, TypeError):
        return DEFAULT_MAX_RETRIES


def get_ranking_weights() -> Dict[str, float]:
    """Retrieve scoring and ranking weights."""
    def _read_float(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, default))
        except (ValueError, TypeError):
            return default

    return {
        "weight_agreement": _read_float("CONSENSUS_WEIGHT_AGREEMENT", DEFAULT_WEIGHT_AGREEMENT),
        "weight_discovery": _read_float("CONSENSUS_WEIGHT_DISCOVERY", DEFAULT_WEIGHT_DISCOVERY),
        "weight_reviewer": _read_float("CONSENSUS_WEIGHT_REVIEWER", DEFAULT_WEIGHT_REVIEWER),
        "weight_hook": _read_float("CONSENSUS_WEIGHT_HOOK", DEFAULT_WEIGHT_HOOK),
        "redundancy_penalty": _read_float("CONSENSUS_PENALTY_REDUNDANCY", DEFAULT_REDUNDANCY_PENALTY),
    }


def validate_consensus_config() -> Tuple[bool, List[str]]:
    """Validate the 4-key consensus configuration at startup.

    Returns:
        (is_valid: bool, issues: List[str])
    """
    issues: List[str] = []
    missing_stages: List[int] = []

    for stage in (1, 2, 3, 4):
        key = get_consensus_api_key(stage)
        if not key:
            missing_stages.append(stage)

    if missing_stages:
        issues.append(
            f"Missing required API key(s) for consensus stage(s): {missing_stages}. "
            "Please configure AI_API_KEY_1, AI_API_KEY_2, AI_API_KEY_3, and AI_API_KEY_4."
        )

    # Check key distinctness (independent keys should ideally not be identical)
    configured_keys = [get_consensus_api_key(s) for s in (1, 2, 3, 4) if get_consensus_api_key(s)]
    if len(configured_keys) == 4 and len(set(configured_keys)) < 4:
        # Warning/note about key independence
        issues.append(
            "Note: Some configured consensus API keys are identical. For true independent "
            "consensus and diverse evaluation, provide distinct API keys for each stage."
        )

    return (len(missing_stages) == 0, issues)


def get_consensus_status_summary() -> Dict[str, Any]:
    """Summary of current consensus configuration for logs and API status."""
    is_valid, issues = validate_consensus_config()
    return {
        "enabled": is_consensus_enabled(),
        "configured": is_valid,
        "strict_mode": is_strict_consensus(),
        "iou_threshold": get_iou_threshold(),
        "semantic_threshold": get_semantic_threshold(),
        "stages": {
            f"stage_{s}": {
                "key_present": get_consensus_api_key(s) is not None,
                "masked_key": mask_key(get_consensus_api_key(s)),
                "model": get_consensus_model(s),
            }
            for s in (1, 2, 3, 4)
        },
        "issues": issues,
    }
