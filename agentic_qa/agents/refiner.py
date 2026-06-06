"""
Refiner Agent — Failure Analysis & Test Improvement.

Analyzes failure patterns from the Judge Agent's verdicts and produces
actionable insights for the Red-Team Agent's next iteration.
"""

import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from agentic_qa.graph.state import QAState
from agentic_qa.utils.prompt_templates import REFINER_SYSTEM_PROMPT, REFINER_ANALYSIS_PROMPT


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        temperature=0.3,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
    )


def _parse_analysis(response_text: str) -> dict:
    try:
        text = response_text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        data = json.loads(text)
        return {
            "new_failure_patterns": data.get("new_failure_patterns", []),
            "root_causes": data.get("root_causes", []),
            "coverage_gaps": data.get("coverage_gaps", []),
            "recommended_focus": data.get("recommended_focus", ""),
            "coverage_score": float(data.get("coverage_score", 0.5)),
            "severity_distribution": data.get("severity_distribution", {}),
        }
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"  Warning: Failed to parse Refiner response: {e}")
        return {
            "new_failure_patterns": ["Parse error - using generic patterns"],
            "root_causes": [],
            "coverage_gaps": [],
            "recommended_focus": "Retry with broader coverage",
            "coverage_score": 0.3,
            "severity_distribution": {},
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
        return {"failure_patterns": ["All tests passed in this iteration"], "coverage_score": 1.0}

    print(f"  Analyzing {len(failures)} failure(s)...")

    verdicts_json = json.dumps(verdicts, indent=2)
    previous_patterns_str = "\n".join(f"  - {p}" for p in previous_patterns) if previous_patterns else "None (first iteration)"

    analysis_prompt = REFINER_ANALYSIS_PROMPT.format(
        iteration=iteration,
        verdicts_json=verdicts_json,
        previous_patterns=previous_patterns_str,
    )

    llm = _get_llm()
    system_prompt = REFINER_SYSTEM_PROMPT.format(
        sut_description=state.get("sut_description", "Unknown System"),
        domain=state.get("domain", "general"),
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=analysis_prompt),
    ]
    response = llm.invoke(messages)
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
    print(f"     Coverage score: {analysis['coverage_score']:.1%}")
    print(f"     Next focus: {analysis['recommended_focus'][:80]}")

    return {
        "failure_patterns": analysis["new_failure_patterns"],
        "coverage_score": analysis["coverage_score"],
    }
