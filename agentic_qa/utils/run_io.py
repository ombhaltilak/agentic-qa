"""
Run persistence: export a run as JSON, and load a saved suite for replay.

A saved run makes results reproducible: the exact generated tests can be
re-executed later (``agentic-qa run --replay run.json``) without the Red-Team
LLM, e.g. to confirm a fix or to gate CI on a fixed suite.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from agentic_qa.golden import golden_metrics
from agentic_qa.policy import policy_stats
from agentic_qa.taxonomy import category_stats, compute_coverage
from agentic_qa.verdicts import is_invalid, scored

SCHEMA_VERSION = 1
_SEVERE = {"critical", "high"}


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo), 4)


def latency_stats(verdicts: list[dict]) -> dict:
    times = [float(v["execution_time"]) for v in verdicts
             if isinstance(v.get("execution_time"), (int, float))]
    return {
        "p50_s": _percentile(times, 0.50),
        "p95_s": _percentile(times, 0.95),
        "max_s": round(max(times), 4) if times else None,
        "samples": len(times),
    }


def run_summary(state: dict) -> dict:
    """Headline metrics for a finished run (used by the CLI gate and JSON export)."""
    all_verdicts = state.get("all_verdicts", []) or []
    verdicts = scored(all_verdicts)  # invalid tests don't count toward quality
    suite = state.get("test_suite", []) or []
    total = len(verdicts)
    passed = sum(1 for v in verdicts if v.get("status") == "pass")
    failing = [v for v in verdicts if v.get("status") != "pass"]
    regression_ids = {tc["id"] for tc in suite if str(tc.get("edge_case_type", "")).startswith("regression_")}
    return {
        "total_tests": len(suite),
        "total_verdicts": len(all_verdicts),
        "invalid_tests": len(all_verdicts) - total,
        "invalid_test_ids": [v.get("test_id") for v in all_verdicts if is_invalid(v)],
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "iteration_pass_rates": state.get("iteration_pass_rates", []),
        "coverage": compute_coverage(suite),
        "critical_or_high_failures": sum(1 for v in failing if str(v.get("severity", "")).lower() in _SEVERE),
        "regression_failures": sum(1 for v in failing if v.get("test_id") in regression_ids),
        "assertion_overrides": sum(1 for v in verdicts if "model_status" in v),
        "needs_human_review": sum(1 for v in verdicts if v.get("needs_human_review")),
        "decided_by": {
            who: sum(1 for v in verdicts if v.get("decided_by") == who) for who in ("jev", "llm")
        },
        "categories": category_stats(suite, all_verdicts),
        "iterations": state.get("current_iteration", 0),
        "sut_latency": latency_stats(all_verdicts),
        "usage": state.get("usage") or _live_usage(),
        "golden": golden_metrics(suite, verdicts),
        "regression_memory": _memory_stats(),
        "policy": policy_stats(state.get("policy"), suite, verdicts),
        "policy_violation_tests": sum(1 for v in verdicts if v.get("policy_violations")),
    }


def _memory_stats() -> dict:
    from agentic_qa.agents.regression_memory import memory_stats

    return memory_stats()


def _live_usage() -> dict:
    from agentic_qa.utils.llm import usage_summary

    return usage_summary()


def save_run(state: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "sut_description": state.get("sut_description"),
        "domain": state.get("domain"),
        "summary": run_summary(state),
        "test_suite": state.get("test_suite", []),
        "verdicts": state.get("all_verdicts", []),
        "failure_patterns": state.get("failure_patterns", []),
        "policy": state.get("policy"),
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def load_test_suite(path: str | Path) -> list[dict]:
    """
    Load test cases for replay from a saved run (``test_suite`` key), a JSON
    list of test cases, or a JSONL file. Each case needs ``input_data``.
    """
    text = Path(path).read_text()
    if str(path).endswith(".jsonl"):
        cases = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        cases = data.get("test_suite", []) if isinstance(data, dict) else data
    from agentic_qa.golden import is_golden_row, normalize_rows

    cases = [c for c in cases if isinstance(c, dict)]
    if cases and all(is_golden_row(c) and "input_data" not in c for c in cases):
        cases = normalize_rows(cases)  # a golden dataset replayed on its own
    cases = [c for c in cases if "input_data" in c]
    if not cases:
        raise ValueError(f"No test cases with 'input_data' found in {path}")
    return cases
