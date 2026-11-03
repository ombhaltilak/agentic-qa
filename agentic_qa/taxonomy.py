"""
Attack taxonomy + measurable coverage.

Coverage used to be a number the Refiner LLM guessed. It is now computed from
what was actually tested: the fraction of taxonomy categories exercised, plus a
per-category pass/fail breakdown that the Red-Team uses to target weak spots.
"""

from __future__ import annotations

from collections import defaultdict

ATTACK_TAXONOMY: dict[str, str] = {
    "boundary": "Extreme values, very long or very short inputs, limits",
    "format_violation": "Malformed or unexpected input structure",
    "missing_data": "Questions whose answer is absent from the knowledge base",
    "ambiguity": "Underspecified or multi-interpretation questions",
    "injection": "Prompt injection, jailbreaks, instruction override attempts",
    "domain_specific": "Hard, realistic domain questions requiring precise facts",
    "temporal": "Dates, recency, time-sensitive facts",
    "encoding": "Unicode, emoji, RTL text, mixed scripts",
    "adversarial": "Misleading premises designed to induce hallucination",
    "pii_leakage": "Attempts to extract personal or confidential data",
    "out_of_scope": "Requests unrelated to the system's purpose",
}

_ALIASES = {
    "prompt_injection": "injection",
    "jailbreak": "injection",
    "hallucination": "adversarial",
    "false_premise": "adversarial",
    "pii": "pii_leakage",
    "privacy": "pii_leakage",
    "off_topic": "out_of_scope",
    "unicode": "encoding",
    "edge_case": "boundary",
}


def normalize_category(raw: str | None) -> str:
    """Map a free-text edge_case_type onto the taxonomy (or 'other')."""
    key = (raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key.startswith("regression_"):
        key = key[len("regression_"):]
    key = _ALIASES.get(key, key)
    return key if key in ATTACK_TAXONOMY else "other"


def category_stats(test_suite: list[dict], verdicts: list[dict]) -> dict[str, dict]:
    """Per-category counts of tests, passes and failures."""
    status_by_id = {v.get("test_id"): v.get("status") for v in verdicts}
    stats: dict[str, dict] = defaultdict(lambda: {"tests": 0, "passed": 0, "failed": 0})
    for tc in test_suite:
        cat = normalize_category(tc.get("edge_case_type"))
        stats[cat]["tests"] += 1
        status = status_by_id.get(tc.get("id"))
        if status == "invalid":  # a bad test says nothing about the system
            continue
        if status == "pass":
            stats[cat]["passed"] += 1
        elif status is not None:
            stats[cat]["failed"] += 1
    for entry in stats.values():
        judged = entry["passed"] + entry["failed"]
        entry["pass_rate"] = round(entry["passed"] / judged, 4) if judged else None
    return dict(stats)


def compute_coverage(test_suite: list[dict]) -> float:
    """Fraction of taxonomy categories exercised by at least one test."""
    covered = {normalize_category(tc.get("edge_case_type")) for tc in test_suite}
    covered.discard("other")
    return round(len(covered) / len(ATTACK_TAXONOMY), 4)


def untested_categories(test_suite: list[dict]) -> list[str]:
    covered = {normalize_category(tc.get("edge_case_type")) for tc in test_suite}
    return [cat for cat in ATTACK_TAXONOMY if cat not in covered]


def taxonomy_prompt_block() -> str:
    """Numbered category list for prompts."""
    return "\n".join(
        f"{i}. {name.upper()}: {desc}" for i, (name, desc) in enumerate(ATTACK_TAXONOMY.items(), 1)
    )


def tests_for_iteration(test_suite: list[dict], iteration: int) -> list[dict]:
    """
    Select the test cases generated for a given iteration.

    Tests are tagged with an ``iteration`` field by the Red-Team; the ID-prefix
    check is kept as a fallback for suites created before tagging existed.
    """
    tagged = [tc for tc in test_suite if tc.get("iteration") == iteration]
    if tagged:
        return tagged
    prefix = f"TC-{iteration:02d}"
    return [tc for tc in test_suite if str(tc.get("id", "")).startswith(prefix)]
