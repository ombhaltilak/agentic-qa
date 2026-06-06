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
    3. No new failure patterns found (convergence)
    
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
    fail_count = sum(1 for v in verdicts if v["status"] in ("fail", "error"))
    if fail_count == 0:
        return "report"
    
    # Condition 4: Check for convergence (same failure patterns repeating)
    failure_patterns = state.get("failure_patterns", [])
    if len(failure_patterns) > 10:
        # If the last 5 patterns are all duplicates of earlier ones, converged
        recent = failure_patterns[-5:]
        earlier = failure_patterns[:-5]
        if all(p in earlier for p in recent):
            return "report"
    
    # Default: Continue refining
    return "refine"
