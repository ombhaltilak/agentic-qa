"""
Routing conditions for the LangGraph QA workflow.

These functions determine whether the system should continue iterating
(refine + re-test) or terminate and generate the final report.
"""

from agentic_qa.graph.state import QAState


def should_continue(state: QAState) -> str:
    """
    Determine whether to continue the QA loop or generate the final report.
    
    Conditions to STOP:
    1. Max iterations reached
    2. All tests passed (100% pass rate in current iteration)
    3. Replay mode (fixed suite runs once)
    4. No new failure patterns found by the last refiner run (convergence)
    
    Returns:
        "refine" - Continue to Refiner Agent for another iteration
        "report" - Stop and generate the final report
    """
    current = state["current_iteration"]
    max_iter = state["max_iterations"]
    verdicts = state.get("judge_verdicts", [])
    
    # Condition 1: Max iterations reached
    if current >= max_iter:
        return "report"
    
    # Condition 2: No verdicts means something went wrong — stop
    if not verdicts:
        return "report"
    
    # Condition 3: All tests passed
    fail_count = sum(1 for v in verdicts if v["status"] in ("fail", "error", "partial"))
    if fail_count == 0:
        return "report"

    # Condition 4: Replay mode runs the fixed suite once
    if state.get("seed_tests"):
        return "report"

    # Condition 5: Convergence — the last refiner run found no new failure patterns
    new_counts = state.get("new_pattern_counts", [])
    if new_counts and new_counts[-1] == 0:
        return "report"

    # Default: Continue refining
    return "refine"
