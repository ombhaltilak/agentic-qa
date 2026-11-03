"""
Re-judge a saved run, and measure verdicts against human labels.

    # same outputs, different judge — no calls to the system under test
    agentic-qa rejudge qa_run.json --judge-model gpt-4o --output qa_run_gpt4o.json
    # how often were the verdicts right?
    agentic-qa label qa_run.json --out labels.jsonl
    agentic-qa agreement qa_run.json qa_run_gpt4o.json --labels labels.jsonl

Re-judging needs the outputs and retrieved contexts stored in the run (runs saved by
v0.3+ include them). Deterministic assertions and policy checks run again too, so the
comparison is the whole judging stack, not just the LLM.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from agentic_qa.calibration import wilson_interval


def _key(text) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _combined_from_run(run: dict) -> list[dict]:
    verdicts = {v.get("test_id"): v for v in run.get("verdicts", [])}
    combined = []
    for tc in run.get("test_suite", []):
        v = verdicts.get(tc.get("id"))
        if not v or "sut_output" not in v:
            continue
        combined.append({
            "test_id": tc["id"],
            "input_data": tc.get("input_data", ""),
            "expected_behavior": tc.get("expected_behavior", ""),
            "edge_case_type": tc.get("edge_case_type", "other"),
            "difficulty": tc.get("difficulty", "medium"),
            "sut_output": v.get("sut_output", ""),
            "retrieved_contexts": v.get("retrieved_contexts", []),
            "execution_error": None,
            "execution_time": v.get("execution_time", 0),
            **{k: tc[k] for k in ("golden", "reference_answer", "answerable", "target_rule") if k in tc},
        })
    return combined


def rejudge_run(run: dict, judge_model: str | None = None, policy: list[dict] | None = None) -> dict:
    """Return a copy of ``run`` with fresh verdicts from the (LLM) judging stack."""
    from agentic_qa.agents.assertions import run_assertions
    from agentic_qa.agents.judge import _run_llm_judge, reconcile_verdicts
    from agentic_qa.policy import policy_assertions
    from agentic_qa.utils import llm as llm_utils
    from agentic_qa.utils.run_io import run_summary

    if judge_model:
        os.environ["JUDGE_MODEL_NAME"] = judge_model
    policy = policy if policy is not None else run.get("policy")
    combined = _combined_from_run(run)
    if not combined:
        raise ValueError("Run has no stored outputs to re-judge")
    llm_utils.reset_usage()
    state = {"sut_description": run.get("sut_description", ""), "domain": run.get("domain", "general"),
             "policy": policy}
    assertion_results = {c["test_id"]: run_assertions(c) + policy_assertions(c, policy) for c in combined}
    llm_verdicts, _ = _run_llm_judge(state, combined)
    verdicts = reconcile_verdicts(combined, llm_verdicts, "llm", {}, assertion_results)
    by_case = {c["test_id"]: c for c in combined}
    for v in verdicts:
        case = by_case[v["test_id"]]
        v["sut_output"] = case["sut_output"]
        v["execution_time"] = case["execution_time"]
        if case.get("retrieved_contexts"):
            v["retrieved_contexts"] = case["retrieved_contexts"]
        v["judge_model"] = llm_utils.model_for_role("judge")
    suite = [tc for tc in run.get("test_suite", []) if tc.get("id") in by_case]
    new = {**run, "test_suite": suite, "verdicts": verdicts, "policy": policy,
           "rejudged_from": run.get("created_at"), "judge_model": llm_utils.model_for_role("judge")}
    new["summary"] = run_summary({"test_suite": suite, "all_verdicts": verdicts, "policy": policy,
                                  "usage": llm_utils.usage_summary()})
    return new


def load_labels(path: str | Path) -> dict[tuple[str, str], str]:
    """(input, output) -> pass|fail. 'bad_test' labels are excluded (the test itself was wrong)."""
    labels = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("label") in ("pass", "fail"):
            labels[(_key(row.get("input_data")), _key(row.get("sut_output")))] = row["label"]
    return labels


def agreement(run: dict, labels: dict[tuple[str, str], str]) -> dict:
    """How well the run's final verdicts match human labels.

    'fail' is the positive class: precision = of the failures reported, how many were real;
    recall = of the real failures, how many were caught.
    """
    tests = {t["id"]: t for t in run.get("test_suite", [])}
    pairs, unmatched = [], 0
    for v in run.get("verdicts", []):
        tc = tests.get(v.get("test_id"), {})
        label = labels.get((_key(tc.get("input_data")), _key(v.get("sut_output"))))
        if label is None:
            unmatched += 1
            continue
        # "invalid" doesn't report a failure, so it counts as a pass prediction
        pred = "pass" if v.get("status") in ("pass", "invalid") else "fail"
        pairs.append((pred, label, v.get("test_id")))
    n = len(pairs)
    tp = sum(p == "fail" and l == "fail" for p, l, _ in pairs)
    fp = sum(p == "fail" and l == "pass" for p, l, _ in pairs)
    fn = sum(p == "pass" and l == "fail" for p, l, _ in pairs)
    correct = sum(p == l for p, l, _ in pairs)
    return {
        "judge_model": run.get("judge_model") or (run.get("verdicts") or [{}])[0].get("judge_model"),
        "labeled": n,
        "unmatched": unmatched,
        "accuracy": round(correct / n, 4) if n else None,
        "accuracy_95ci": wilson_interval(correct, n),
        "failure_precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "failure_precision_95ci": wilson_interval(tp, tp + fp),
        "failure_recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "failure_recall_95ci": wilson_interval(tp, tp + fn),
        "invalid_verdicts": sum(1 for v in run.get("verdicts", []) if v.get("status") == "invalid"),
        "false_alarms": [t for p, l, t in pairs if p == "fail" and l == "pass"],
        "missed_failures": [t for p, l, t in pairs if p == "pass" and l == "fail"],
    }
