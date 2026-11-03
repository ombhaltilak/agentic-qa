"""
Red-Team Agent — Adversarial Test Case Generator.

Generates diverse, targeted adversarial test inputs for the System Under Test (SUT),
targeting edge cases, boundary conditions, and common vulnerabilities in RAG/LLM systems.

On iteration 1, it generates broad-spectrum tests. On subsequent iterations,
it uses failure patterns from the Refiner Agent to generate increasingly
targeted and sophisticated test cases.
"""

import os
from langchain_core.messages import SystemMessage, HumanMessage
from agentic_qa.graph.state import QAState
from agentic_qa.utils.prompt_templates import (
    RED_TEAM_SYSTEM_PROMPT,
    RED_TEAM_GENERATION_PROMPT,
    RED_TEAM_REFINEMENT_CONTEXT,
)
from agentic_qa.agents.regression_memory import invalid_examples, replay_cases_for_iteration
from agentic_qa.golden import GOLDEN_FIELDS
from agentic_qa.policy import red_team_block, rule_attack_prompt, rules_to_target
from agentic_qa.taxonomy import category_stats, normalize_category, untested_categories
from agentic_qa.utils import llm as llm_utils


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
    
    context = RED_TEAM_REFINEMENT_CONTEXT.format(
        failure_patterns="\n".join(f"  - {p}" for p in failure_patterns[-10:]),
        total_tests=total_tests,
        total_failures=total_failures,
        top_failure_categories=top_str or "None identified yet",
    )
    return context + _coverage_guidance(state)


def _coverage_guidance(state: QAState) -> str:
    """Point the generator at untested and weakest taxonomy categories."""
    suite = state.get("test_suite", [])
    missing = untested_categories(suite)
    stats = category_stats(suite, state.get("all_verdicts", []))
    weak = sorted(
        (c for c, s in stats.items() if c != "other" and s["pass_rate"] is not None and s["pass_rate"] < 1.0),
        key=lambda c: stats[c]["pass_rate"],
    )[:3]
    lines = []
    if missing:
        lines.append(f"UNTESTED CATEGORIES (cover these first): {', '.join(missing)}")
    if weak:
        lines.append("WEAKEST CATEGORIES (probe deeper): " + ", ".join(
            f"{c} ({stats[c]['pass_rate']:.0%} pass)" for c in weak))
    return ("\n" + "\n".join(lines) + "\n") if lines else ""


def _parse_test_cases(response_text: str, iteration: int) -> list:
    """Parse LLM response into structured test cases. Raises ValueError on bad JSON."""
    data = llm_utils.parse_json_response(response_text)
    raw_cases = data.get("test_cases", []) if isinstance(data, dict) else []
    normalized = []
    for tc in raw_cases:
        if not isinstance(tc, dict) or not str(tc.get("input_data", "")).strip():
            continue
        normalized.append({
            # IDs are always assigned here: LLM-supplied IDs can collide across iterations.
            "id": f"TC-{iteration:02d}{len(normalized) + 1:02d}",
            "iteration": iteration,
            "input_data": str(tc.get("input_data", "")),
            "expected_behavior": str(tc.get("expected_behavior", "")),
            "edge_case_type": normalize_category(tc.get("edge_case_type")),
            "difficulty": str(tc.get("difficulty", "medium")).lower(),
            "rationale": str(tc.get("rationale", "")),
            **({"target_rule": str(tc["target_rule"])} if tc.get("target_rule") else {}),
        })
    if not normalized:
        raise ValueError("Response contained no usable test cases")
    return normalized


def _fallback_cases(iteration: int) -> list:
    return [{
        "id": f"TC-{iteration:02d}01",
        "iteration": iteration,
        "input_data": "",
        "expected_behavior": "Should handle empty input gracefully (ask for clarification, no crash)",
        "edge_case_type": "boundary",
        "difficulty": "easy",
        "rationale": "Fallback test case: red-team generation failed",
    }]


def _seeded_cases(seed_tests: list, iteration: int, tag: str = "S") -> list:
    """Re-tag a saved/fixed (or golden) test suite; golden fields are preserved."""
    return [
        {
            **{k: tc[k] for k in GOLDEN_FIELDS if k in tc},
            "id": f"TC-{iteration:02d}{tag}{i:02d}",
            "iteration": iteration,
            "input_data": str(tc.get("input_data", "")),
            "expected_behavior": str(tc.get("expected_behavior", "")),
            "edge_case_type": normalize_category(tc.get("edge_case_type")),
            "difficulty": str(tc.get("difficulty", "medium")),
            "rationale": tc.get("rationale") or f"Replayed from {tc.get('id', 'saved suite')}",
        }
        for i, tc in enumerate(seed_tests, start=1)
    ]


def _invalid_block(state: dict | None = None) -> str:
    bad = invalid_examples()
    suite = {tc.get("id"): tc for tc in (state or {}).get("test_suite", [])}
    for v in (state or {}).get("all_verdicts", []):
        tc = suite.get(v.get("test_id"))
        if v.get("status") == "invalid" and tc:
            bad.append({"input_data": tc.get("input_data"), "expected_behavior": tc.get("expected_behavior"),
                        "invalid_reason": v.get("invalid_reason", "")})
    bad = bad[-10:]
    if not bad:
        return ""
    lines = ["", "A human marked these earlier tests INVALID (their expected behavior was wrong for this "
             "system). Do not write tests like them:"]
    for c in bad:
        why = f" — {c['invalid_reason']}" if c.get("invalid_reason") else ""
        lines.append(f"- {str(c.get('input_data', ''))[:160]!r} expected {str(c.get('expected_behavior', ''))[:120]!r}{why}")
    return "\n".join(lines) + "\n"


def _keep_valid_targets(cases: list, rules: list | None) -> None:
    """Drop target_rule values that aren't real rule ids (so coverage stats stay honest)."""
    ids = {r["id"] for r in rules or []}
    for tc in cases:
        if tc.get("target_rule") and tc["target_rule"] not in ids:
            tc.pop("target_rule")


def _fill_rule_gaps(missing: list, iteration: int, start: int, system_prompt: str) -> list:
    """One follow-up call for required rules the generator skipped; target_rule is set by us."""
    llm = llm_utils.get_llm("red_team", temperature=0.9)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=rule_attack_prompt(missing))]
    try:
        response = llm_utils.invoke(llm, messages, "red_team")
        raw = llm_utils.parse_json_response(response.content).get("test_cases", [])
    except Exception as e:
        print(f"  ⚠️  Rule-targeted generation failed: {e}")
        return []
    pending = [r["id"] for r in missing]
    cases = []
    for tc in raw if isinstance(raw, list) else []:
        if not pending or not isinstance(tc, dict) or not str(tc.get("input_data", "")).strip():
            continue
        rid = tc.get("target_rule") if tc.get("target_rule") in pending else pending[0]
        pending.remove(rid)
        cases.append({
            "id": f"TC-{iteration:02d}{start + len(cases):02d}",
            "iteration": iteration,
            "input_data": str(tc["input_data"]),
            "expected_behavior": str(tc.get("expected_behavior", "")),
            "edge_case_type": normalize_category(tc.get("edge_case_type")),
            "difficulty": str(tc.get("difficulty", "hard")).lower(),
            "rationale": str(tc.get("rationale", "")) or f"Targets policy rule {rid}",
            "target_rule": rid,
        })
    return cases


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
    
    if state.get("seed_tests"):
        test_cases = _seeded_cases(state["seed_tests"], iteration)
        print(f"  📼 Replay mode: running {len(test_cases)} saved test case(s) (no generation)")
        return {"test_suite": test_cases, "current_iteration": iteration}

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
                "iteration": iteration,
            })
        return {
            "test_suite": test_cases,
            "current_iteration": iteration,
        }

    
    # Build the prompt
    policy = state.get("policy")
    required = rules_to_target(policy, state.get("test_suite", []),
                               int(os.getenv("POLICY_TESTS_PER_ITERATION", "2")))
    failure_context = _build_failure_context(state) + red_team_block(policy, required) + _invalid_block(state)
    
    generation_prompt = RED_TEAM_GENERATION_PROMPT.format(
        num_tests=num_tests,
        sut_description=state.get("sut_description", "A RAG/LLM system"),
        domain=state.get("domain", "general"),
        iteration=iteration,
        iteration_prefix=f"{iteration:02d}",
        failure_context=failure_context,
    )
    
    # Call the LLM — format system prompt with domain context
    llm = llm_utils.get_llm("red_team", temperature=0.9)
    system_prompt = RED_TEAM_SYSTEM_PROMPT.format(
        sut_description=state.get("sut_description", "Unknown System"),
        domain=state.get("domain", "general"),
        sut_architecture=state.get("sut_architecture", "Architecture unknown. Treat as black-box.")
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=generation_prompt),
    ]
    
    # One retry on malformed output before falling back.
    test_cases = None
    for attempt in range(2):
        try:
            response = llm_utils.invoke(llm, messages, "red_team")
            test_cases = _parse_test_cases(response.content, iteration)
            break
        except Exception as e:
            print(f"  ⚠️  Red-Team generation attempt {attempt + 1} failed: {e}")
    if test_cases is None:
        test_cases = _fallback_cases(iteration)

    # Every iteration attacks its assigned policy rules, even if the generator ignored them.
    _keep_valid_targets(test_cases, policy)
    covered = {tc.get("target_rule") for tc in test_cases}
    missing = [r for r in required if r["id"] not in covered]
    if missing:
        extra = _fill_rule_gaps(missing, iteration, len(test_cases) + 1, system_prompt)
        test_cases += extra
        print(f"  🎯 Added {len(extra)} rule-targeted test(s): {', '.join(tc['target_rule'] for tc in extra)}")

    # Golden items (reference answers) run once, in the first iteration.
    if iteration == 1 and state.get("golden_tests"):
        golden = _seeded_cases(state["golden_tests"], iteration, tag="G")
        test_cases = golden + test_cases
        print(f"  🥇 Added {len(golden)} golden case(s) with reference answers")

    # Replay prior-run failures from regression memory (once, at the start of a run).
    replay_limit = int(os.getenv("REGRESSION_MEMORY_REPLAY", "3"))
    regression_cases = replay_cases_for_iteration(iteration, limit=replay_limit) if iteration == 1 else []
    if regression_cases:
        test_cases = regression_cases + test_cases
        print(f"  ♻️  Added {len(regression_cases)} regression replay case(s) from memory")
    
    print(f"  ✅ Generated {len(test_cases)} test cases:")
    for tc in test_cases:
        target = f" → {tc['target_rule']}" if tc.get("target_rule") else ""
        print(f"     [{tc['id']}] {tc['edge_case_type']:20s} | {tc['difficulty']:6s} | {tc['input_data'][:60]}...{target}")
    
    return {
        "test_suite": test_cases,
        "current_iteration": iteration,
    }
