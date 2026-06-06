"""
Agentic QA - Main Public API

This file exposes the simple `run_autonomous_test` function, allowing developers 
to test their RAG systems in just a few lines of code.
"""

import os
from typing import Callable, Optional

# Expose SUT adapters for developers if they want advanced setups
from agentic_qa.sut.base import BaseSUTAdapter
from agentic_qa.sut.api_adapter import APIAdapter
from agentic_qa.sut.callable_adapter import CallableAdapter
from agentic_qa.sut import set_active_sut
from agentic_qa.utils.summary import display_qa_results

# Import the core workflow
from agentic_qa.graph.workflow import build_qa_graph, get_initial_state


def run_autonomous_test(
    target_function: Optional[Callable] = None,
    api_endpoint: Optional[str] = None,
    system_name: str = "Target System",
    system_description: str = "A generic RAG system",
    domain: str = "general",
    max_iterations: int = 3,
    tests_per_iteration: int = 5,
    model_name: str = "gpt-4o-mini",
) -> dict:
    """
    Run an autonomous multi-agent QA test against a target system.
    
    You must provide EITHER a `target_function` (a python function) OR 
    an `api_endpoint` (a URL string).
    
    Args:
        target_function: A python function that takes a string query and returns a string answer.
        api_endpoint: A URL endpoint (e.g., http://localhost:8000/chat) to test.
        system_name: The name of the system being tested.
        system_description: A description of what the system does. Highly important for agents!
        domain: The domain of the system (e.g., 'financial', 'healthcare', 'customer support').
        max_iterations: How many times the agents should refine and retry their tests.
        tests_per_iteration: How many tests the Red-Team agent generates per round.
        model_name: The LLM to use for the agents (default: gpt-4o-mini).
        
    Returns:
        A dictionary containing the final execution state, including the test suite, verdicts,
        failure patterns, and the final Markdown report.
    """
    # 1. Configure Environment variables needed by LangGraph
    os.environ["MAX_ITERATIONS"] = str(max_iterations)
    os.environ["TESTS_PER_ITERATION"] = str(tests_per_iteration)
    os.environ["MODEL_NAME"] = model_name
    
    # 2. Setup the SUT Adapter
    if target_function:
        adapter = CallableAdapter(
            fn=target_function,
            description=system_description,
            system_name=system_name,
            domain=domain
        )
    elif api_endpoint:
        adapter = APIAdapter(
            endpoint=api_endpoint,
            description=system_description,
            system_name=system_name,
            domain=domain
        )
    else:
        raise ValueError("You must provide either a `target_function` or an `api_endpoint`.")

    # Register the adapter globally for the Executor Agent to use
    set_active_sut(adapter)
    
    # 3. Build and Run the Graph
    print(f"🚀 Starting Autonomous QA Test against: {system_name}")
    print(f"Domain: {domain} | Max Iterations: {max_iterations}")
    
    graph = build_qa_graph()
    initial_state = get_initial_state()
    initial_state["max_iterations"] = max_iterations
    
    final_state = None
    # Stream the graph to provide real-time console feedback
    for event in graph.stream(initial_state, stream_mode="values"):
        final_state = event
        
    print("\n✅ Autonomous QA Test Complete!")
    print(f"Coverage Score: {final_state.get('coverage_score', 0):.1%}")
    print(f"Total Failure Patterns Found: {len(final_state.get('failure_patterns', []))}")
    
    return final_state


# Define what is exported when someone runs `from agentic_qa import *`
__all__ = [
    "run_autonomous_test",
    "APIAdapter",
    "CallableAdapter",
    "BaseSUTAdapter",
    "display_qa_results"
]
