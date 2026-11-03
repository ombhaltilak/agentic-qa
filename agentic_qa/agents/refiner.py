"""
Refiner Agent — Failure Analysis & Test Improvement.

Analyzes failure patterns from the Judge Agent's verdicts and produces
actionable insights for the Red-Team Agent's next iteration.
"""

import json

from langchain_core.messages import SystemMessage, HumanMessage

from agentic_qa.graph.state import QAState
from agentic_qa.utils import llm as llm_utils
from agentic_qa.utils.prompt_templates import REFINER_SYSTEM_PROMPT, REFINER_ANALYSIS_PROMPT


def _parse_analysis(response_text: str) -> dict:
    try:
        data = llm_utils.parse_json_response(response_text)
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        patterns = [str(p) for p in data.get("new_failure_patterns", []) if str(p).strip()]
        return {
            "new_failure_patterns": patterns,
            "root_causes": data.get("root_causes", []),
            "coverage_gaps": data.get("coverage_gaps", []),
            "recommended_focus": str(data.get("recommended_focus", "")),
        }
    except ValueError as e:
        print(f"  Warning: Failed to parse Refiner response: {e}")
        return {
            "new_failure_patterns": [],
            "root_causes": [],
            "coverage_gaps": [],
            "recommended_focus": "Retry with broader coverage",
        }


def refiner_node(state: QAState) -> dict:
    """LangGraph node: Refiner Agent. Analyzes failures and produces insights."""
    iteration = state.get("current_iteration", 1)
    verdicts = state.get("judge_verdicts", [])
    previous_patterns = state.get("failure_patterns", [])

    print(f"\n{'='*60}")
    print(f"  REFINER AGENT - Iteration {iteration}")
    print(f"{'='*60}")

    failures = [v for v in verdicts if v["status"] in ("fail", "error", "partial")]
    if not failures:
        print(f"  No failures to analyze! All tests passed.")
        return {}

    print(f"  Analyzing {len(failures)} failure(s)...")

    verdicts_json = json.dumps(verdicts, indent=2)
    previous_patterns_str = "\n".join(f"  - {p}" for p in previous_patterns) if previous_patterns else "None (first iteration)"

    analysis_prompt = REFINER_ANALYSIS_PROMPT.format(
        iteration=iteration,
        verdicts_json=verdicts_json,
        previous_patterns=previous_patterns_str,
    )

    llm = llm_utils.get_llm("refiner", temperature=0.3)
    system_prompt = REFINER_SYSTEM_PROMPT.format(
        sut_description=state.get("sut_description", "Unknown System"),
        domain=state.get("domain", "general"),
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=analysis_prompt),
    ]
    response = llm_utils.invoke(llm, messages, "refiner")
    analysis = _parse_analysis(response.content)

    print(f"\n  Analysis Results:")
    print(f"     New failure patterns: {len(analysis['new_failure_patterns'])}")
    for p in analysis["new_failure_patterns"]:
        print(f"       - {p}")
    if analysis["root_causes"]:
        print(f"     Root causes:")
        for rc in analysis["root_causes"]:
            print(f"       > {rc}")
    if analysis["coverage_gaps"]:
        print(f"     Coverage gaps:")
        for gap in analysis["coverage_gaps"]:
            print(f"       o {gap}")
    print(f"     Next focus: {analysis['recommended_focus'][:80]}")

    # Only genuinely new patterns are accumulated (keeps convergence detection meaningful).
    seen = {p.strip().lower() for p in previous_patterns}
    new_patterns = [p for p in analysis["new_failure_patterns"] if p.strip().lower() not in seen]
    return {"failure_patterns": new_patterns, "new_pattern_counts": [len(new_patterns)]}
