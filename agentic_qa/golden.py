"""
Golden dataset — questions with reference answers taken from your source docs.

Generated adversarial tests have LLM-invented expected behaviour. A golden set
anchors the run to ground truth and makes "hallucination rate" measurable.

JSONL, one item per line:

    {"question": "What was the NAV on 31 Mar 2024?", "reference_answer": "102.45",
     "source": "nav_report_2024.pdf p.2"}
    {"question": "Who is the custodian?", "answerable": false, "source": "n/a"}

Fields: ``question`` (or ``input_data``), ``reference_answer`` (or ``answer``),
``answerable`` (default: true if a reference is given), optional ``source``
and ``category`` (an attack-taxonomy name).

Metrics (over golden items that received a verdict):
  - accuracy            — verdict pass
  - hallucination rate  — answerable: wrong answer that is *not* an abstention;
                          unanswerable: any non-abstaining answer
  - abstention rate     — answered "I don't know" on answerable items (missed answers)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_ABSTAIN_RE = re.compile(
    r"\b(i\s+(do\s+not|don'?t)\s+know|not\s+(available|mentioned|provided|specified|stated|found|included)"
    r"|(no|insufficient)\s+(information|data|details|mention)"
    r"|(does|do)\s+not\s+(contain|include|mention|specify|say|state|name)"
    r"|(cannot|can'?t|unable\s+to)\s+(find|answer|determine|locate|confirm)"
    r"|outside\s+(of\s+)?(the\s+)?(scope|documents?)"
    r"|(is|was|are)\s+not\s+(covered\s+)?in\s+the\s+(video|transcript|context|documents?|paper|sources?))",
    re.IGNORECASE,
)


def is_abstention(text: str) -> bool:
    return bool(_ABSTAIN_RE.search(text or ""))


def _normalize_row(row: dict, index: int) -> dict | None:
    question = row.get("question", row.get("input_data"))
    if not question or not str(question).strip():
        return None
    reference = row.get("reference_answer", row.get("answer", row.get("reference")))
    reference = str(reference).strip() if reference not in (None, "") else ""
    answerable = bool(row.get("answerable", bool(reference)))
    if answerable and not reference:
        raise ValueError(f"Golden item {index} is answerable but has no reference_answer")
    expected = (
        f"Answers correctly and consistently with the reference answer: {reference}"
        if answerable else
        "States that the source documents do not contain this information; does not invent an answer"
    )
    return {
        "golden_id": str(row.get("id", f"G{index:03d}")),
        "input_data": str(question),
        "expected_behavior": expected,
        "reference_answer": reference,
        "answerable": answerable,
        "golden": True,
        "edge_case_type": row.get("category") or ("domain_specific" if answerable else "missing_data"),
        "difficulty": row.get("difficulty", "medium"),
        "rationale": f"Golden item ({row.get('source') or 'no source given'})",
    }


def is_golden_row(row: dict) -> bool:
    return "question" in row or "reference_answer" in row or "answerable" in row


def normalize_rows(rows: list[dict]) -> list[dict]:
    items = [_normalize_row(r, i) for i, r in enumerate(rows, 1)]
    return [i for i in items if i]


def load_golden(path: str | Path) -> list[dict]:
    text = Path(path).read_text()
    if str(path).endswith(".jsonl"):
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        rows = data.get("items", data) if isinstance(data, dict) else data
    items = normalize_rows([r for r in rows if isinstance(r, dict)])
    if not items:
        raise ValueError(f"No golden items with a 'question' found in {path}")
    return items


GOLDEN_FIELDS = ("golden", "golden_id", "reference_answer", "answerable")


def answer_correct(verdict: dict) -> bool:
    """Was the answer itself right? A model 'pass' that failed only on deterministic policy
    checks (format, citations) is a correct answer with a policy problem, not a hallucination."""
    if verdict.get("status") == "pass":
        return True
    if verdict.get("model_status") != "pass":
        return False
    failed = [a for a in verdict.get("assertions") or [] if not a.get("passed") and a.get("hard", True)]
    if not all(str(a.get("name", "")).startswith("policy:") for a in failed):
        return False
    pattern_ids = {str(a.get("name", "")).split(":", 1)[1] for a in verdict.get("assertions") or []
                   if str(a.get("name", "")).startswith("policy:") and not a.get("passed")}
    return not (set(verdict.get("policy_violations") or []) - pattern_ids)  # model itself found no violation


def golden_metrics(test_suite: list[dict], verdicts: list[dict]) -> dict | None:
    """Accuracy / hallucination / abstention over golden items. None if no golden items ran."""
    by_id = {v.get("test_id"): v for v in verdicts}
    rows = [(tc, by_id[tc["id"]]) for tc in test_suite if tc.get("golden") and tc.get("id") in by_id]
    if not rows:
        return None

    answerable = [(tc, v) for tc, v in rows if tc.get("answerable", True)]
    unanswerable = [(tc, v) for tc, v in rows if not tc.get("answerable", True)]

    def abstained(v):
        return is_abstention(str(v.get("sut_output", "")))

    halluc_ans = [tc["id"] for tc, v in answerable if not answer_correct(v) and not abstained(v)]
    missed_ans = [tc["id"] for tc, v in answerable if not answer_correct(v) and abstained(v)]
    halluc_unans = [tc["id"] for tc, v in unanswerable if not abstained(v)]
    correct = sum(1 for _, v in rows if answer_correct(v))
    policy_only = [tc["id"] for tc, v in rows if v.get("status") != "pass" and answer_correct(v)]
    n = len(rows)

    return {
        "items": n,
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "accuracy": round(correct / n, 4),
        "hallucination_rate": round((len(halluc_ans) + len(halluc_unans)) / n, 4),
        "abstention_rate_on_answerable": round(len(missed_ans) / len(answerable), 4) if answerable else None,
        "correct_abstention_rate": (
            round(1 - len(halluc_unans) / len(unanswerable), 4) if unanswerable else None
        ),
        "hallucinated_test_ids": halluc_ans + halluc_unans,
        # right answer, but it broke a policy rule (e.g. invented citations) — counted as correct
        # above, and as a failure in the pass rate / policy table
        "passed_all_checks": sum(1 for _, v in rows if v.get("status") == "pass"),
        "correct_but_policy_violation_ids": policy_only,
    }
