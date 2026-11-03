"""
Executor Agent — Generic System Under Test Runner.

Executes test cases against ANY connected SUT (RAG, API, function).
Uses the active SUT adapter from the registry — works with:
  - Built-in Financial RAG demo
  - Any RAG connected via API endpoint
  - Any Python function wrapped as a callable

Tests run concurrently (EXECUTOR_MAX_WORKERS, default 4) with a per-test
timeout (EXECUTOR_TIMEOUT seconds, default 60).
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from agentic_qa.graph.state import QAState
from agentic_qa.sut import get_active_sut
from agentic_qa.taxonomy import tests_for_iteration
from agentic_qa.agents.local_evaluator import _flatten_contexts

_CONTEXT_KEYS = ("retrieved_contexts", "contexts", "context", "source_documents", "documents", "sources")


def _extract_adapter_contexts(output: dict) -> list[str]:
    """First non-empty context field, in priority order. Fields are not merged: e.g. an
    adapter's "sources" are citation labels, not the retrieved text, and would duplicate it."""
    for key in _CONTEXT_KEYS:
        if key in output:
            contexts = [c.strip() for c in _flatten_contexts(output[key]) if c and c.strip()]
            if contexts:
                return contexts
    return []


def _run_one(sut, tc: dict) -> dict:
    """Execute a single test case. Never raises."""
    start = time.time()
    try:
        output = sut.process(tc["input_data"])
        elapsed = time.time() - start
        if not isinstance(output, dict):
            output = {"status": "success", "output": str(output)}
        status = output.get("status", "success")
        answer = output.get("output", output.get("sut_output", ""))
        return {
            "test_id": tc["id"],
            "sut_output": str(answer),
            "retrieved_contexts": _extract_adapter_contexts(output),
            "execution_time": round(elapsed, 4),
            "error": (output.get("error") or str(answer)) if status == "error" else None,
        }
    except Exception as e:  # adapter bugs must not crash the pipeline
        return {
            "test_id": tc["id"],
            "sut_output": "",
            "retrieved_contexts": [],
            "execution_time": round(time.time() - start, 4),
            "error": f"{type(e).__name__}: {e}",
        }


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

    current_tests = tests_for_iteration(test_suite, iteration)
    if not current_tests:
        n = int(os.getenv("TESTS_PER_ITERATION", "5"))
        current_tests = test_suite[-n:]

    sut = get_active_sut()
    workers = max(1, int(os.getenv("EXECUTOR_MAX_WORKERS", "4")))
    timeout = float(os.getenv("EXECUTOR_TIMEOUT", "60"))
    print(f"  SUT: {sut.name}")
    print(f"  Executing {len(current_tests)} test cases ({workers} worker(s), {timeout:.0f}s timeout)...")

    execution_results: list[dict] = []
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [(tc, pool.submit(_run_one, sut, tc)) for tc in current_tests]
        for tc, fut in futures:
            try:
                result = fut.result(timeout=timeout)
            except FutureTimeout:
                result = {
                    "test_id": tc["id"],
                    "sut_output": "",
                    "retrieved_contexts": [],
                    "execution_time": timeout,
                    "error": f"Timeout: SUT did not respond within {timeout:.0f}s",
                }
            status = "ERROR" if result["error"] else "OK"
            print(f"  ▶ [{tc['id']}] {status} ({result['execution_time']:.3f}s)")
            execution_results.append(result)
    finally:
        # Don't block on hung SUT calls; abandon them.
        pool.shutdown(wait=False, cancel_futures=True)

    errors = sum(1 for r in execution_results if r["error"])
    avg_time = sum(r["execution_time"] for r in execution_results) / max(len(execution_results), 1)
    print(f"\n  📊 Execution Summary:")
    print(f"     Tests executed: {len(execution_results)}")
    print(f"     Errors/crashes: {errors}")
    print(f"     Avg exec time:  {avg_time:.3f}s")

    return {
        "execution_results": execution_results,
    }
