"""
Regression memory for failed QA cases.

Stores compact failed cases from previous runs and replays them at the start
of later runs. Intentionally local and dependency-free.

A single failure is not trusted (LLM judges and SUTs can be flaky):

  suspected  failed once — replayed next run to get a second observation
  confirmed  failed on two or more observations — a real regression
  flaky      was suspected, then passed on replay — no longer replayed
  fixed      was confirmed, then passed — no longer replayed

Replay prioritises confirmed cases. ``--fail-on-regression`` counts replayed
cases that fail again, i.e. at least two observed failures.
"""

from __future__ import annotations

import datetime
import json
import os
import re
from pathlib import Path

ACTIVE = ("confirmed", "suspected")
CONFIRM_AFTER = 2


def memory_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_") or "default"


def memory_path() -> Path:
    """Resolved at call time so tests / the CLI can redirect it.

    Each system under test gets its own file (REGRESSION_MEMORY_SCOPE, set from the
    system name), so failures from one app are never replayed against another.
    REGRESSION_MEMORY_PATH overrides this with an explicit file.
    """
    explicit = os.getenv("REGRESSION_MEMORY_PATH")
    if explicit:
        return Path(explicit)
    scope = os.getenv("REGRESSION_MEMORY_SCOPE")
    if scope:
        return Path("reports/regression_memory") / f"{memory_slug(scope)}.json"
    return Path("reports/regression_memory.json")


def _status(case: dict) -> str:
    """Status with backward compatibility for files written before statuses existed."""
    if case.get("status"):
        return case["status"]
    if case.get("last_status") == "pass":
        return "fixed"
    return "confirmed" if case.get("recurrence_count", 1) >= CONFIRM_AFTER else "suspected"


def load_regression_cases(limit: int | None = None, include_fixed: bool = True) -> list[dict]:
    path = memory_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    cases = data.get("failed_cases", []) if isinstance(data, dict) else []
    if not isinstance(cases, list):
        return []
    if not include_fixed:
        cases = [c for c in cases if _status(c) in ACTIVE]
    return cases[-limit:] if limit else cases


def replay_cases_for_iteration(iteration: int, limit: int = 3) -> list[dict]:
    if limit <= 0:
        return []
    active = load_regression_cases(include_fixed=False)
    # Confirmed first, then most recent suspects.
    ordered = [c for c in reversed(active) if _status(c) == "confirmed"] + \
              [c for c in reversed(active) if _status(c) == "suspected"]
    replayed = []
    for index, case in enumerate(ordered[:limit], start=1):
        replayed.append({
            "id": f"TC-{iteration:02d}R{index:02d}",
            "input_data": case.get("input_data", ""),
            "expected_behavior": case.get("expected_behavior", ""),
            "edge_case_type": f"regression_{_base_category(case.get('edge_case_type'))}",
            "difficulty": case.get("difficulty", "high"),
            "iteration": iteration,
            "regression_status": _status(case),
            "rationale": (f"Regression replay ({_status(case)}, failed {case.get('recurrence_count', 1)}x) "
                          f"from {case.get('original_test_id', 'unknown')}"),
        })
    return replayed


def _base_category(raw: str | None) -> str:
    cat = str(raw or "failure")
    while cat.startswith("regression_"):
        cat = cat[len("regression_"):]
    return cat or "failure"


def remember_failures(combined_results: list[dict], verdicts: list[dict]) -> int:
    """Update memory from this iteration's verdicts. Returns cases added/updated as failing."""
    failing_ids = {
        verdict.get("test_id")
        for verdict in verdicts
        if verdict.get("status") in {"fail", "error", "partial"}
    }
    existing = load_regression_cases()
    # Tests a human marked invalid (wrong expectation) are never re-remembered or replayed.
    invalid_inputs = {_norm(c.get("input_data")) for c in existing if _status(c) == "invalid"}
    by_key = {
        (case.get("input_data", ""), case.get("expected_behavior", "")): case
        for case in existing
    }
    verdict_map = {verdict.get("test_id"): verdict for verdict in verdicts}

    changed = 0
    touched = False
    now = datetime.datetime.now().isoformat()
    sut_version = os.getenv("SUT_VERSION", "unknown")

    for result in combined_results:
        key = (result.get("input_data", ""), result.get("expected_behavior", ""))
        verdict = verdict_map.get(result.get("test_id"), {})
        if _norm(result.get("input_data")) in invalid_inputs:
            continue
        if verdict.get("status") == "invalid":
            # The judge found the test itself wrong: retire a remembered copy instead of replaying it.
            if key in by_key and _status(by_key[key]) != "invalid":
                case = by_key[key]
                case["status"] = "invalid"
                case["invalid_reason"] = "auto (judge): " + str(verdict.get("invalid_reason") or "")[:200]
                case["invalidated_date"] = now
                touched = True
            continue

        if result.get("test_id") not in failing_ids:
            # Passing now: a suspect was flaky; a confirmed regression is fixed.
            if verdict.get("status") == "pass" and key in by_key and _status(by_key[key]) in ACTIVE:
                case = by_key[key]
                case["status"] = "flaky" if _status(case) == "suspected" else "fixed"
                case["last_status"] = "pass"
                case["fixed_date"] = now
                touched = True
            continue

        existing_case = by_key.get(key, {})
        # A flaky/fixed case that fails again starts counting from its history.
        recurrence_count = existing_case.get("recurrence_count", 0) + 1
        by_key[key] = {
            "original_test_id": existing_case.get("original_test_id") or result.get("test_id", "unknown"),
            "input_data": result.get("input_data", ""),
            "expected_behavior": result.get("expected_behavior", ""),
            "edge_case_type": _base_category(result.get("edge_case_type")),
            "difficulty": result.get("difficulty", "medium"),
            "status": "confirmed" if recurrence_count >= CONFIRM_AFTER else "suspected",
            "last_status": verdict.get("status", "unknown"),
            "failure_category": verdict.get("failure_category"),
            "severity": verdict.get("severity", "medium"),
            "reasoning": verdict.get("reasoning", "")[:500],
            "first_failed_date": existing_case.get("first_failed_date", now),
            "last_failed_date": now,
            "recurrence_count": recurrence_count,
            "sut_version": sut_version,
            "fixed_date": None,
        }
        changed += 1

    if not changed and not touched:
        return 0
    _save(list(by_key.values()))
    return changed


def _norm(text) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _save(cases: list[dict]) -> None:
    path = memory_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"failed_cases": cases[-200:]}, indent=2))


def invalidate_cases(selector: str, reason: str = "") -> list[dict]:
    """Mark remembered tests as invalid (a bad test, not a bug in the system).

    ``selector`` is a 1-based index from ``agentic-qa memory list`` or text contained in
    the test input. Invalid tests are not replayed, not re-remembered when they fail again,
    and the Red-Team is told not to write similar ones.
    """
    cases = load_regression_cases()
    if str(selector).isdigit():
        i = int(selector) - 1
        targets = [cases[i]] if 0 <= i < len(cases) else []
    else:
        needle = _norm(selector)
        targets = [c for c in cases if needle and needle in _norm(c.get("input_data"))]
    now = datetime.datetime.now().isoformat()
    for c in targets:
        c["status"] = "invalid"
        c["invalid_reason"] = reason or "marked invalid by a human"
        c["invalidated_date"] = now
    if targets:
        _save(cases)
    return targets


def invalid_examples(limit: int = 8) -> list[dict]:
    """Recently invalidated tests, for steering the Red-Team away from them."""
    return [c for c in load_regression_cases() if _status(c) == "invalid"][-limit:]


def memory_stats() -> dict:
    counts = {"suspected": 0, "confirmed": 0, "flaky": 0, "fixed": 0, "invalid": 0}
    for case in load_regression_cases():
        counts[_status(case)] = counts.get(_status(case), 0) + 1
    return counts
