"""
LangGraph Workflow — Multi-Agent QA Pipeline.

Defines the graph structure connecting all 5 agents:
  Red-Team -> Executor -> Judge -> [Refiner -> loop | Reporter -> end]

Uses LangGraph's StateGraph with conditional edges for the
continue/stop decision after each Judge evaluation.
"""

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from agentic_qa.graph.state import QAState
from agentic_qa.graph.conditions import should_continue
from agentic_qa.agents.red_team import red_team_node
from agentic_qa.agents.executor import executor_node
from agentic_qa.agents.judge import judge_node
from agentic_qa.agents.refiner import refiner_node
from agentic_qa.agents.reporter import reporter_node
from agentic_qa.agents.discovery import discovery_node

load_dotenv()


def build_qa_graph() -> StateGraph:
    """
    Build and compile the LangGraph QA workflow.
    
    Graph structure:
        red_team -> executor -> judge -> conditional_edge
            if "refine" -> refiner -> red_team (loop)
            if "report" -> reporter -> END
    
    Returns:
        Compiled LangGraph StateGraph
    """
    # Create the state graph
    workflow = StateGraph(QAState)
    
    # ── Add nodes ──
    workflow.add_node("discovery", discovery_node)
    workflow.add_node("red_team", red_team_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("refiner", refiner_node)
    workflow.add_node("reporter", reporter_node)
    
    # ── Define edges ──
    # Linear flow: Discovery -> Red-Team -> Executor -> Judge
    workflow.add_edge("discovery", "red_team")
    workflow.add_edge("red_team", "executor")
    workflow.add_edge("executor", "judge")
    
    # Conditional: After Judge, either Refine (loop) or Report (end)
    workflow.add_conditional_edges(
        "judge",
        should_continue,
        {
            "refine": "refiner",
            "report": "reporter",
        }
    )
    
    # Refiner loops back to Red-Team for next iteration
    workflow.add_edge("refiner", "red_team")
    
    # Reporter terminates the graph
    workflow.add_edge("reporter", END)
    
    # ── Set entry point ──
    workflow.set_entry_point("discovery")
    
    # ── Compile ──
    graph = workflow.compile()
    
    return graph


def get_initial_state(sut_description: str = None, domain: str = None) -> QAState:
    """
    Create the initial state for a QA run.
    
    Args:
        sut_description: Description of the SUT (auto-detected from active adapter if None)
        domain: Domain of the SUT (auto-detected from active adapter if None)
    
    Returns:
        Initial QAState with SUT description and config
    """
    from agentic_qa.sut import get_active_sut
    
    active_sut = get_active_sut()
    
    return {
        "sut_description": sut_description or active_sut.describe(),
        "sut_architecture": None,
        "domain": domain or active_sut.domain,
        "test_suite": [],
        "current_iteration": 0,
        "max_iterations": int(os.getenv("MAX_ITERATIONS", "3")),
        "execution_results": [],
        "judge_verdicts": [],
        "failure_patterns": [],
        "all_verdicts": [],
        "coverage_score": 0.0,
        "iteration_pass_rates": [],
        "final_report": None,
    }


def run_qa_pipeline() -> dict:
    """
    Execute the full QA pipeline.
    
    Returns:
        Final state after all iterations complete
    """
    print("\n" + "=" * 70)
    print("  MULTI-AGENT AUTONOMOUS QA SYSTEM")
    print("  LangGraph Pipeline Starting...")
    print("=" * 70)
    
    graph = build_qa_graph()
    initial_state = get_initial_state()
    
    print(f"\n  Config:")
    print(f"    Max iterations: {initial_state['max_iterations']}")
    print(f"    Domain: {initial_state['domain']}")
    print(f"    LLM Model: {os.getenv('MODEL_NAME', 'gpt-4o-mini')}")
    print(f"    LangSmith: {'Enabled' if os.getenv('LANGCHAIN_TRACING_V2') == 'true' else 'Disabled'}")
    
    # Execute the graph
    final_state = graph.invoke(initial_state)
    
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    
    # Print final metrics
    total_tests = len(final_state.get("test_suite", []))
    total_verdicts = len(final_state.get("all_verdicts", []))
    pass_rates = final_state.get("iteration_pass_rates", [])
    coverage = final_state.get("coverage_score", 0.0)
    
    print(f"\n  Final Metrics:")
    print(f"    Total test cases generated: {total_tests}")
    print(f"    Total evaluations: {total_verdicts}")
    print(f"    Iteration pass rates: {[f'{r:.1%}' for r in pass_rates]}")
    print(f"    Final coverage score: {coverage:.1%}")
    print(f"    Iterations completed: {final_state.get('current_iteration', 0)}")
    
    return final_state
