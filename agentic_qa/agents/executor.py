"""
Executor Agent — Generic System Under Test Runner.

Executes test cases against ANY connected SUT (RAG, API, function).
Uses the active SUT adapter from the registry — works with:
  - Built-in Financial RAG demo
  - Any RAG connected via API endpoint
  - Any Python function wrapped as a callable
"""

import time
from agentic_qa.graph.state import QAState
from agentic_qa.sut import get_active_sut


def executor_node(state: QAState) -> dict:
    """
    LangGraph node: Executor Agent.
    
    Runs each test case from the current iteration through whatever
    SUT is currently active and collects results.
    """
    iteration = state.get("current_iteration", 1)
    test_suite = state.get("test_suite", [])
    
    print(f"\n{'='*60}")
    print(f"⚡ EXECUTOR AGENT — Iteration {iteration}")
    print(f"{'='*60}")
    
    # Get test cases for the current iteration only
    iter_prefix = f"TC-{iteration:02d}"
    current_tests = [tc for tc in test_suite if tc["id"].startswith(iter_prefix)]
    
    if not current_tests:
        num_per_iter = 5
        current_tests = test_suite[-num_per_iter:]
    
    # Get the active SUT (whatever the user connected)
    sut = get_active_sut()
    print(f"  SUT: {sut.name}")
    print(f"  Executing {len(current_tests)} test cases...")
    
    execution_results = []
    
    for tc in current_tests:
        test_id = tc["id"]
        input_data = tc["input_data"]
        
        print(f"  ▶ Running [{test_id}]...", end=" ")
        
        start_time = time.time()
        try:
            output = sut.process(input_data)
            exec_time = time.time() - start_time
            
            # Normalize output — adapters return "output" key
            sut_output = output.get("output", output.get("sut_output", str(output)))
            status = output.get("status", "unknown")
            
            result = {
                "test_id": test_id,
                "sut_output": str(output),
                "execution_time": round(exec_time, 4),
                "error": output.get("error") if status == "error" else None,
            }
            print(f"Done ({exec_time:.3f}s) — Status: {status}")
            
        except Exception as e:
            exec_time = time.time() - start_time
            result = {
                "test_id": test_id,
                "sut_output": "",
                "execution_time": round(exec_time, 4),
                "error": str(e),
            }
            print(f"ERROR ({exec_time:.3f}s) — {e}")
        
        execution_results.append(result)
    
    errors = sum(1 for r in execution_results if r["error"])
    avg_time = sum(r["execution_time"] for r in execution_results) / max(len(execution_results), 1)
    print(f"\n  📊 Execution Summary:")
    print(f"     Tests executed: {len(execution_results)}")
    print(f"     Errors/crashes: {errors}")
    print(f"     Avg exec time:  {avg_time:.3f}s")
    
    return {
        "execution_results": execution_results,
    }
