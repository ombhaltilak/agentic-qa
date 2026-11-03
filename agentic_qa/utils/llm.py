"""
Shared LLM + JSON helpers used by every agent.

Centralising model construction lets us:
  - use a different (ideally stronger) model for the Judge than for the
    Red-Team, so the system is not grading its own homework (JUDGE_MODEL_NAME)
  - apply the same timeout / retry policy everywhere
  - swap the model in tests with a single monkeypatch of ``get_llm``
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Any

# Roles map to the env var that may override the model for that role.
_ROLE_MODEL_ENV = {
    "judge": "JUDGE_MODEL_NAME",
}


# Reviewing whether a generated test's expectation is right needs a strong model: in our
# measurements gpt-4o-mini accepted every expectation, including clearly wrong ones.
DEFAULT_VALIDITY_MODEL = "gpt-4o"


def model_for_role(role: str) -> str:
    """Resolve the model name for an agent role."""
    if role == "validity":
        return os.getenv("JUDGE_VALIDITY_MODEL") or DEFAULT_VALIDITY_MODEL
    default = os.getenv("MODEL_NAME", "gpt-4o-mini")
    env_key = _ROLE_MODEL_ENV.get(role)
    if env_key:
        return os.getenv(env_key) or default
    return default


def get_llm(role: str, temperature: float):
    """Build a chat model for the given agent role."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_for_role(role),
        temperature=temperature,
        timeout=float(os.getenv("LLM_TIMEOUT", "120")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
    )


# ── Token / cost ledger ────────────────────────────────────────────────────
# USD per 1M tokens (input, output). Estimates — verify against your provider's
# current pricing and override with LLM_PRICES_JSON='{"model": [in, out], ...}'.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "jev-1.13-free": (0.0, 0.0),
    "jev-1.13": (0.042, 0.0),
    "jev-latest": (0.042, 0.0),
}

_ledger_lock = threading.Lock()
_ledger: list[dict] = []


def _prices() -> dict[str, tuple[float, float]]:
    prices = dict(DEFAULT_PRICES)
    raw = os.getenv("LLM_PRICES_JSON")
    if raw:
        try:
            prices.update({k: (float(v[0]), float(v[1])) for k, v in json.loads(raw).items()})
        except (ValueError, TypeError, IndexError, json.JSONDecodeError):
            pass
    return prices


def record_usage(role: str, model: str, input_tokens: int, output_tokens: int) -> None:
    with _ledger_lock:
        _ledger.append({"role": role, "model": model or "unknown",
                        "input_tokens": int(input_tokens or 0), "output_tokens": int(output_tokens or 0)})


def reset_usage() -> None:
    with _ledger_lock:
        _ledger.clear()


def usage_summary() -> dict:
    """Token totals and estimated USD cost, overall and per agent role."""
    prices = _prices()
    with _ledger_lock:
        rows = list(_ledger)
    by_role: dict[str, dict] = {}
    unpriced: set[str] = set()
    total_cost = 0.0
    for r in rows:
        price = prices.get(r["model"])
        cost = 0.0
        if price is None:
            unpriced.add(r["model"])
        else:
            cost = (r["input_tokens"] * price[0] + r["output_tokens"] * price[1]) / 1_000_000
        total_cost += cost
        agg = by_role.setdefault(r["role"], {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
        agg["calls"] += 1
        agg["input_tokens"] += r["input_tokens"]
        agg["output_tokens"] += r["output_tokens"]
        agg["cost_usd"] += cost
    for agg in by_role.values():
        agg["cost_usd"] = round(agg["cost_usd"], 6)
    return {
        "calls": len(rows),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
        "estimated_cost_usd": round(total_cost, 6),
        "by_role": by_role,
        "unpriced_models": sorted(unpriced),
    }


def invoke(llm, messages, role: str):
    """Call a chat model and record its token usage under ``role``."""
    response = llm.invoke(messages)
    usage = getattr(response, "usage_metadata", None) or {}
    model = getattr(llm, "model_name", None) or model_for_role(role)
    record_usage(role, model, usage.get("input_tokens", 0), usage.get("output_tokens", 0))
    return response


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_response(text: str) -> Any:
    """
    Parse JSON out of an LLM response.

    Handles markdown code fences and leading/trailing prose by falling back to
    the outermost ``{...}`` block. Raises ``ValueError`` if nothing parses.
    """
    text = (text or "").strip()
    candidates = [text]
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.insert(0, fence.group(1).strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    raise ValueError("No valid JSON object found in LLM response")
