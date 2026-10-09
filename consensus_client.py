"""Reliable, schema-enforced client caller for each consensus stage.

Features:
  - Uses the stage's assigned API key without key-swapping/rotation
  - Strict schema enforcement via Pydantic
  - Bounded retries with exponential backoff and jitter
  - Explicit timeouts
  - Clear distinction between transient errors and permanent authentication/quota errors
  - Latency and token usage tracking
  - Never logs or prints raw API keys
"""
from __future__ import annotations

import json
import random
import time
from typing import Any, Dict, Optional, Tuple, Type

from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, ValidationError

import consensus_config
import gemini_worker
import llm_backend


class ConsensusAuthError(Exception):
    """Raised when an API key fails authentication or lacks permission."""
    pass


class ConsensusQuotaError(Exception):
    """Raised when an API key has exhausted its quota."""
    pass


class ConsensusStageExecutionError(Exception):
    """Raised when an AI stage cannot complete within retry bounds."""
    pass


def is_permanent_error(message: str) -> Tuple[bool, Optional[str]]:
    """Determine if an error is permanent (auth, permission, quota) and should not be retried."""
    msg_upper = message.upper()
    if any(tok in msg_upper for tok in ("API_KEY_INVALID", "INVALID_API_KEY", "PERMISSION_DENIED", "UNAUTHENTICATED", "401", "403")):
        return True, "Authentication / Permission error. Verify that the configured API key is valid and has Gemini API enabled."
    if any(tok in msg_upper for tok in ("BILLING_NOT_ACTIVE", "ACCOUNT_DISABLED")):
        return True, "Account billing or quota disabled error."
    return False, None


def is_transient_error(message: str) -> bool:
    """Determine if an error is temporary and eligible for backoff retry."""
    msg = message.lower()
    return any(tok in msg for tok in (
        "503", "unavailable", "429", "resource_exhausted",
        "500", "internal", "overloaded", "deadline", "timeout",
        "empty response body", "did not contain a json object",
        "failed to parse gemini json response", "connecterror",
        "readtimeout", "remoteprotocolerror", "502", "504"
    ))


def execute_consensus_call(
    stage: int,
    prompt: str,
    response_schema: Type[BaseModel],
    max_retries: Optional[int] = None,
    timeout_seconds: Optional[float] = None,
) -> Tuple[BaseModel, Dict[str, Any]]:
    """Execute a model call for a specific consensus stage.

    Args:
        stage: The stage number (1, 2, 3, or 4).
        prompt: The text prompt to execute.
        response_schema: The Pydantic model for structured output validation.
        max_retries: Max retry attempts (defaults to consensus_config).
        timeout_seconds: Timeout per attempt.

    Returns:
        (validated_pydantic_instance, telemetry_dict)

    Raises:
        ConsensusAuthError: On API key authentication failure.
        ConsensusStageExecutionError: If stage fails after max retries.
    """
    api_key = consensus_config.get_consensus_api_key(stage)
    model_name = consensus_config.get_consensus_model(stage)
    masked_key = consensus_config.mask_key(api_key)

    if not api_key and not llm_backend.active():
        raise ConsensusAuthError(
            f"Stage {stage} requires an API key, but none was found. "
            f"Please set AI_API_KEY_{stage} (or GEMINI_API_KEY_{stage})."
        )

    retries = max_retries if max_retries is not None else consensus_config.get_max_retries()
    timeout = timeout_seconds if timeout_seconds is not None else consensus_config.get_timeout_seconds()

    start_time = time.time()
    last_error: Optional[Exception] = None
    telemetry: Dict[str, Any] = {
        "stage": stage,
        "model": model_name,
        "masked_key": masked_key,
        "attempts": 0,
        "latency_seconds": 0.0,
        "tokens": {},
        "cost_usd": None,
    }

    use_local = llm_backend.active() and not api_key

    for attempt in range(1, retries + 1):
        telemetry["attempts"] = attempt
        attempt_start = time.time()

        try:
            if use_local:
                parsed_dict, cost = llm_backend.generate_json(prompt, response_schema, model=model_name)
                validated = response_schema.model_validate(parsed_dict)
                telemetry["latency_seconds"] = round(time.time() - start_time, 3)
                return validated, telemetry

            client = genai.Client(api_key=api_key)
            config = genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
            )

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )

            # Check for safety / policy blocks
            gemini_worker.raise_if_blocked(response)

            # Parse and validate response
            parsed_obj = getattr(response, "parsed", None)
            if parsed_obj is not None:
                if isinstance(parsed_obj, response_schema):
                    validated = parsed_obj
                elif hasattr(parsed_obj, "model_dump"):
                    validated = response_schema.model_validate(parsed_obj.model_dump())
                else:
                    validated = response_schema.model_validate(parsed_obj)
            else:
                raw_text = gemini_worker._get_response_text(response)
                parsed_json = gemini_worker._parse_json_response_text(raw_text)
                validated = response_schema.model_validate(parsed_json)

            # Record telemetry
            cost_info = gemini_worker._calculate_cost_analysis(response, model_name)
            if cost_info:
                telemetry["cost_usd"] = cost_info.get("total_cost")
                telemetry["tokens"] = {
                    "input": cost_info.get("input_tokens"),
                    "output": cost_info.get("output_tokens"),
                    "thinking": cost_info.get("thinking_tokens"),
                }

            telemetry["latency_seconds"] = round(time.time() - start_time, 3)
            return validated, telemetry

        except gemini_worker.GeminiBlockedError as e:
            # Deterministic policy block — retrying won't change policy
            telemetry["latency_seconds"] = round(time.time() - start_time, 3)
            raise ConsensusStageExecutionError(f"Stage {stage} blocked by content policy: {e}") from e

        except ValidationError as e:
            last_error = e
            # Schema validation issue from model output
            if attempt >= retries:
                telemetry["latency_seconds"] = round(time.time() - start_time, 3)
                raise ConsensusStageExecutionError(
                    f"Stage {stage} output schema validation failed after {attempt} attempts: {e}"
                ) from e
            wait = 2.0 * attempt + random.uniform(0.1, 1.0)
            time.sleep(wait)

        except Exception as e:
            err_msg = str(e)
            last_error = e

            # Check for permanent auth/quota errors
            is_perm, perm_desc = is_permanent_error(err_msg)
            if is_perm:
                telemetry["latency_seconds"] = round(time.time() - start_time, 3)
                raise ConsensusAuthError(
                    f"Stage {stage} (key {masked_key}) permanent failure: {perm_desc} Details: {err_msg[:200]}"
                ) from e

            # Transient error handling with exponential backoff & jitter
            if attempt >= retries or not is_transient_error(err_msg):
                telemetry["latency_seconds"] = round(time.time() - start_time, 3)
                raise ConsensusStageExecutionError(
                    f"Stage {stage} failed on attempt {attempt}/{retries}: {err_msg[:250]}"
                ) from e

            backoff = min(30.0, (2.0 ** (attempt - 1)) * 3.0) + random.uniform(0.2, 1.5)
            time.sleep(backoff)

    telemetry["latency_seconds"] = round(time.time() - start_time, 3)
    raise ConsensusStageExecutionError(
        f"Stage {stage} could not complete after {retries} attempts. Last error: {last_error}"
    )
