"""
Red-Team Agent — Adversarial Test Case Generator.

Generates diverse, targeted adversarial test inputs for the System Under Test (SUT),
targeting edge cases, boundary conditions, and common vulnerabilities in RAG/LLM systems.

On iteration 1, it generates broad-spectrum tests. On subsequent iterations,
it uses failure patterns from the Refiner Agent to generate increasingly
targeted and sophisticated test cases.
"""

import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from agentic_qa.graph.state import QAState
from agentic_qa.utils.prompt_templates import (
    RED_TEAM_SYSTEM_PROMPT,
    RED_TEAM_GENERATION_PROMPT,
    RED_TEAM_REFINEMENT_CONTEXT,
)
from agentic_qa.agents.regression_memory import replay_cases_for_iteration


def _get_llm() -> ChatOpenAI:
    """Initialize the LLM for the Red-Team Agent."""
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        temperature=0.9,  # High creativity for diverse adversarial inputs
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
    )


def _build_failure_context(state: QAState) -> str:
    """Build context from previous failures to guide targeted generation."""
    failure_patterns = state.get("failure_patterns", [])
    all_verdicts = state.get("all_verdicts", [])
    
    if not failure_patterns and not all_verdicts:
        return "This is the FIRST iteration. Generate a diverse initial test suite covering all edge case categories."
    
    # Count failures
    total_tests = len(all_verdicts)
    total_failures = sum(1 for v in all_verdicts if v.get("status") in ("fail", "error"))
    
    # Find top failure categories
    categories = {}
    for v in all_verdicts:
        if v.get("status") in ("fail", "error"):
            cat = v.get("failure_category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
    
    top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]
    top_str = ", ".join([f"{cat} ({count})" for cat, count in top_categories])
    
    return RED_TEAM_REFINEMENT_CONTEXT.format(
        failure_patterns="\n".join(f"  - {p}" for p in failure_patterns[-10:]),
        total_tests=total_tests,
        total_failures=total_failures,
        top_failure_categories=top_str or "None identified yet",
    )


def _parse_test_cases(response_text: str, iteration: int) -> list:
    """Parse LLM response into structured test cases."""
    try:
        # Try to extract JSON from the response
        # Handle cases where LLM wraps JSON in markdown code blocks
        text = response_text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        data = json.loads(text)
        test_cases = data.get("test_cases", [])
        
        # Validate and normalize each test case
        normalized = []
        for i, tc in enumerate(test_cases):
            normalized.append({
                "id": tc.get("id", f"TC-{iteration:02d}{i+1:02d}"),
                "input_data": tc.get("input_data", ""),
                "expected_behavior": tc.get("expected_behavior", ""),
                "edge_case_type": tc.get("edge_case_type", "adversarial"),
                "difficulty": tc.get("difficulty", "medium"),
                "rationale": tc.get("rationale", ""),
            })
        
        return normalized
        
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"  ⚠️  Failed to parse Red-Team response: {e}")
        # Fallback: generate a single default test case
        return [{
            "id": f"TC-{iteration:02d}01",
            "input_data": "Test with empty input",
            "expected_behavior": "Should handle empty or minimal input gracefully",
            "edge_case_type": "missing_data",
            "difficulty": "medium",
            "rationale": "Fallback test case due to parsing error",
        }]


def red_team_node(state: QAState) -> dict:
    """
    LangGraph node: Red-Team Agent.
    
    Generates adversarial test cases for the current iteration.
    Uses failure patterns from previous iterations to create more targeted tests.
    
    Args:
        state: Current QAState
        
    Returns:
        State update with new test_suite entries and incremented iteration
    """
    iteration = state.get("current_iteration", 0) + 1
    num_tests = int(os.getenv("TESTS_PER_ITERATION", "5"))
    
    print(f"\n{'='*60}")
    print(f"🔴 RED-TEAM AGENT — Iteration {iteration}")
    print(f"{'='*60}")
    print(f"  Generating {num_tests} adversarial test cases...")
    
    if state.get("benchmark_mode"):
        print("  📊 Benchmark mode active: using fixed dataset...")
        test_cases = []
        for i in range(num_tests):
            test_cases.append({
                "id": f"TC-BM-{iteration:02d}{i+1:02d}",
                "input_data": f"Benchmark query {i+1}",
                "expected_behavior": f"Expected standard behavior for benchmark query {i+1}",
                "edge_case_type": "benchmark",
                "difficulty": "medium",
                "rationale": "Fixed benchmark dataset case",
            })
        return {
            "test_suite": test_cases,
            "current_iteration": iteration,
        }

    
    # Build the prompt
    failure_context = _build_failure_context(state)
    
    generation_prompt = RED_TEAM_GENERATION_PROMPT.format(
        num_tests=num_tests,
        sut_description=state.get("sut_description", "A RAG/LLM system"),
        domain=state.get("domain", "general"),
        iteration=iteration,
        iteration_prefix=f"{iteration:02d}",
        failure_context=failure_context,
    )
    
    # Call the LLM — format system prompt with domain context
    llm = _get_llm()
    system_prompt = RED_TEAM_SYSTEM_PROMPT.format(
        sut_description=state.get("sut_description", "Unknown System"),
        domain=state.get("domain", "general"),
        sut_architecture=state.get("sut_architecture", "Architecture unknown. Treat as black-box.")
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=generation_prompt),
    ]
    
    response = llm.invoke(messages)
    
    # Parse test cases from LLM response and replay prior failures.
    test_cases = _parse_test_cases(response.content, iteration)
    replay_limit = int(os.getenv("REGRESSION_MEMORY_REPLAY", "3"))
    regression_cases = replay_cases_for_iteration(iteration, limit=replay_limit)
    if regression_cases:
        test_cases = regression_cases + test_cases
        print(f"  ♻️  Added {len(regression_cases)} regression replay case(s) from memory")
    
    print(f"  ✅ Generated {len(test_cases)} test cases:")
    for tc in test_cases:
        print(f"     [{tc['id']}] {tc['edge_case_type']:20s} | {tc['difficulty']:6s} | {tc['input_data'][:60]}...")
    
    return {
        "test_suite": test_cases,
        "current_iteration": iteration,
    }
