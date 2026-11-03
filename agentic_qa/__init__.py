"""
Agentic QA - Main Public API

This file exposes the simple `run_autonomous_test` function, allowing developers
to test their RAG systems in just a few lines of code.
"""

import os
from typing import Callable, Optional

from agentic_qa.sut.base import BaseSUTAdapter
from agentic_qa.sut.api_adapter import APIAdapter, OpenAIChatAdapter
from agentic_qa.sut.langchain_adapter import LangChainAdapter
from agentic_qa.sut.callable_adapter import CallableAdapter
from agentic_qa.sut import set_active_sut
from agentic_qa.utils.summary import display_qa_results
from agentic_qa.utils.run_io import load_test_suite, run_summary, save_run
from agentic_qa.golden import load_golden
from agentic_qa.policy import load_policy
from agentic_qa.agents.assertions import AssertionResult, register_assertion
from agentic_qa.graph.workflow import build_qa_graph, get_initial_state

__version__ = "0.3.0"


def _is_runnable(obj) -> bool:
    """A LangChain Runnable (has .invoke) rather than a plain function."""
    import inspect

    return hasattr(obj, "invoke") and not (inspect.isfunction(obj) or inspect.ismethod(obj))


def run_autonomous_test(
    target_function: Optional[Callable] = None,
    api_endpoint: Optional[str] = None,
    adapter: Optional[BaseSUTAdapter] = None,
    system_name: str = "Target System",
    system_description: str = "A generic RAG system",
    domain: str = "general",
    max_iterations: int = 3,
    tests_per_iteration: int = 5,
    model_name: str = "gpt-4o-mini",
    judge_model: Optional[str] = None,
    judge_backend: str = "auto",
    replay_file: Optional[str] = None,
    dataset_file: Optional[str] = None,
    policy_file: Optional[str] = None,
    enable_ragas: bool = False,
    max_workers: int = 4,
    save_path: Optional[str] = None,
) -> dict:
    """
    Run an autonomous multi-agent QA test against a target system.

    Provide ONE of `target_function`, `api_endpoint`, or a custom `adapter`.

    Args:
        target_function: A python function that takes a string query and returns a string
            (or a dict with "output"/"answer" and optional "contexts").
        api_endpoint: A URL endpoint (e.g., http://localhost:8000/chat) to test.
        adapter: A custom BaseSUTAdapter instance (advanced).
        system_name: The name of the system being tested.
        system_description: What the system does. Highly important for the agents!
        domain: Domain of the system (e.g. 'financial', 'healthcare', 'customer support').
        max_iterations: How many generate → execute → judge → refine rounds to run.
        tests_per_iteration: How many tests the Red-Team agent generates per round.
        model_name: LLM used by the agents (default: gpt-4o-mini).
        judge_model: Separate (ideally stronger) model for the LLM judge.
        judge_backend: "llm", "jev", "cascade", "ensemble", or "auto" (cascade when
            Jev is configured via OPENCODE_API_KEY or TYPESAFE_API_KEY, else llm).
        replay_file: Re-run a saved suite (run JSON / JSON list / JSONL) instead of generating.
            A golden dataset can be replayed on its own this way.
        dataset_file: Golden dataset (JSONL of questions + reference answers from your docs),
            run in the first iteration alongside generated tests; enables hallucination rate.
        policy_file: YAML/JSON policy of must / must_not rules (see agentic_qa.policy).
        enable_ragas: Also compute RAGAS metrics (extra LLM cost).
        max_workers: Concurrent SUT calls in the executor.
        save_path: Where to write the run JSON (default: reports/qa_run_iter<N>.json).

    Returns:
        The final state: test suite, verdicts, failure patterns, metrics and Markdown report.
        `run_summary(final_state)` gives headline metrics.
    """
    os.environ["MAX_ITERATIONS"] = str(max_iterations)
    os.environ["TESTS_PER_ITERATION"] = str(tests_per_iteration)
    os.environ["MODEL_NAME"] = model_name
    os.environ["JUDGE_BACKEND"] = judge_backend
    os.environ["ENABLE_RAGAS"] = "1" if enable_ragas else "0"
    os.environ["EXECUTOR_MAX_WORKERS"] = str(max_workers)
    if judge_model:
        os.environ["JUDGE_MODEL_NAME"] = judge_model

    if sum(x is not None for x in (target_function, api_endpoint, adapter)) != 1:
        raise ValueError("Provide exactly one of `target_function`, `api_endpoint` or `adapter`.")
    if target_function is not None and _is_runnable(target_function):
        adapter = LangChainAdapter(target_function, description=system_description,
                                   system_name=system_name, domain=domain)
    elif target_function is not None:
        adapter = CallableAdapter(fn=target_function, description=system_description,
                                  system_name=system_name, domain=domain)
    elif api_endpoint is not None:
        adapter = APIAdapter(endpoint=api_endpoint, description=system_description,
                             system_name=system_name, domain=domain)
    set_active_sut(adapter)
    # Keep regression memory per system so one app's failures aren't replayed against another
    os.environ["REGRESSION_MEMORY_SCOPE"] = adapter.name

    seed_tests = load_test_suite(replay_file) if replay_file else None
    golden_tests = load_golden(dataset_file) if dataset_file else None
    policy = load_policy(policy_file) if policy_file else None

    print(f"🚀 Starting Autonomous QA Test against: {adapter.name}")
    print(f"Domain: {domain} | Max Iterations: {max_iterations}"
          + (f" | Replay: {len(seed_tests)} saved tests" if seed_tests else "")
          + (f" | Golden: {len(golden_tests)} items" if golden_tests else "")
          + (f" | Policy: {len(policy)} rules" if policy else ""))

    graph = build_qa_graph()
    initial_state = get_initial_state(seed_tests=seed_tests, golden_tests=golden_tests, policy=policy)
    initial_state["max_iterations"] = 1 if seed_tests else max_iterations

    final_state = None
    # Stream the graph to provide real-time console feedback
    for event in graph.stream(initial_state, stream_mode="values"):
        final_state = event

    if save_path:
        save_run(final_state, save_path)

    summary = run_summary(final_state)
    print("\n✅ Autonomous QA Test Complete!")
    print(f"Pass rate: {summary['pass_rate']:.1%} | Coverage: {summary['coverage']:.1%} "
          f"| Critical/high failures: {summary['critical_or_high_failures']}"
          + (f" | Hallucination rate: {summary['golden']['hallucination_rate']:.1%}" if summary["golden"] else ""))
    return final_state


__all__ = [
    "run_autonomous_test",
    "run_summary",
    "save_run",
    "load_test_suite",
    "load_golden",
    "load_policy",
    "register_assertion",
    "AssertionResult",
    "APIAdapter",
    "OpenAIChatAdapter",
    "LangChainAdapter",
    "CallableAdapter",
    "BaseSUTAdapter",
    "display_qa_results",
]
