"""
Regression memory for failed QA cases.

Stores compact failed cases from previous runs and replays them in later
red-team iterations. This is intentionally local and dependency-free.
"""

from __future__ import annotations

import json
import os
import datetime
from pathlib import Path


MEMORY_PATH = Path(os.getenv("REGRESSION_MEMORY_PATH", "reports/regression_memory.json"))


def load_regression_cases(limit: int | None = None) -> list[dict]:
    if not MEMORY_PATH.exists():
        return []
    try:
        data = json.loads(MEMORY_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    cases = data.get("failed_cases", []) if isinstance(data, dict) else []
    if not isinstance(cases, list):
        return []
    return cases[-limit:] if limit else cases


def replay_cases_for_iteration(iteration: int, limit: int = 3) -> list[dict]:
    cases = load_regression_cases(limit=limit)
    replayed = []
    for index, case in enumerate(cases, start=1):
        replayed.append({
            "id": f"TC-{iteration:02d}R{index:02d}",
            "input_data": case.get("input_data", ""),
            "expected_behavior": case.get("expected_behavior", ""),
            "edge_case_type": f"regression_{case.get('edge_case_type', 'failure')}",
            "difficulty": case.get("difficulty", "high"),
            "rationale": f"Regression replay from prior failed case {case.get('original_test_id', 'unknown')}",
        })
    return replayed


def remember_failures(combined_results: list[dict], verdicts: list[dict]) -> int:
    failing_ids = {
        verdict.get("test_id")
        for verdict in verdicts
        if verdict.get("status") in {"fail", "error", "partial"}
    }
    if not failing_ids:
        return 0

    existing = load_regression_cases()
    by_key = {
        (case.get("input_data", ""), case.get("expected_behavior", "")): case
        for case in existing
    }
    verdict_map = {verdict.get("test_id"): verdict for verdict in verdicts}

    added = 0
    now = datetime.datetime.now().isoformat()
    sut_version = os.getenv("SUT_VERSION", "unknown")

    for result in combined_results:
        key = (result.get("input_data", ""), result.get("expected_behavior", ""))
        verdict = verdict_map.get(result.get("test_id"), {})
        is_failing = result.get("test_id") in failing_ids
        
        # If the case existed previously but now passes, we might want to mark it fixed
        # For simplicity, we just keep failing cases.
        if not is_failing:
            if key in by_key and by_key[key].get("last_status") != "pass":
                by_key[key]["fixed_date"] = now
                by_key[key]["last_status"] = "pass"
            continue
            
        existing_case = by_key.get(key, {})
        first_failed_date = existing_case.get("first_failed_date", now)
        recurrence_count = existing_case.get("recurrence_count", 0) + 1

        by_key[key] = {
            "original_test_id": result.get("test_id", "unknown"),
            "input_data": result.get("input_data", ""),
            "expected_behavior": result.get("expected_behavior", ""),
            "edge_case_type": result.get("edge_case_type", "unknown"),
            "difficulty": result.get("difficulty", "medium"),
            "last_status": verdict.get("status", "unknown"),
            "failure_category": verdict.get("failure_category"),
            "severity": verdict.get("severity", "medium"),
            "reasoning": verdict.get("reasoning", "")[:500],
            "first_failed_date": first_failed_date,
            "last_failed_date": now,
            "recurrence_count": recurrence_count,
            "sut_version": sut_version,
            "fixed_date": None,
        }
        added += 1

    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    cases = list(by_key.values())[-200:]
    MEMORY_PATH.write_text(json.dumps({"failed_cases": cases}, indent=2))
    return added
