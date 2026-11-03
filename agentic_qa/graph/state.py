"""
LangGraph State Schema for the Multi-Agent QA System.

Defines the shared state (QAState) that flows between all agent nodes
in the LangGraph workflow. Each agent reads from and writes to this
shared state, enabling fully autonomous multi-agent coordination.
"""

from typing import TypedDict, List, Optional, Annotated
from operator import add


class TestCase(TypedDict, total=False):
    """A single test case in the pipeline state."""
    id: str
    iteration: int
    input_data: str
    expected_behavior: str
    edge_case_type: str
    difficulty: str
    rationale: str


class TestResult(TypedDict):
    """Result from executing a test case against the SUT."""
    test_id: str
    sut_output: str
    retrieved_contexts: List[str]
    execution_time: float
    error: Optional[str]


class Verdict(TypedDict, total=False):
    """Judge Agent's evaluation of a test result."""
    test_id: str
    status: str  # "pass", "fail", "error", "partial", "invalid" (bad test)
    reasoning: str
    severity: str
    failure_category: Optional[str]
    confidence: float


class QAState(TypedDict):
    """
    Shared state for the LangGraph QA workflow.
    
    This state is passed between all agent nodes. Each agent reads
    the fields it needs and updates the fields it owns.
    """
    # ── System Configuration ──
    sut_name: Optional[str]
    sut_description: str
    sut_architecture: Optional[str]
    domain: str  # e.g., "academic", "customer support", "legal", "general"
    benchmark_mode: Optional[bool]
    seed_tests: Optional[List[dict]]  # fixed suite to replay instead of generating
    golden_tests: Optional[List[dict]]  # reference-answer items run in iteration 1
    policy: Optional[List[dict]]  # organisation rules (agentic_qa.policy)
    
    # ── Test Suite (accumulated across iterations) ──
    test_suite: Annotated[List[TestCase], add]
    
    # ── Current Iteration Tracking ──
    current_iteration: int
    max_iterations: int
    
    # ── Execution Results (current iteration) ──
    execution_results: List[TestResult]
    
    # ── Judge Verdicts (current iteration) ──
    judge_verdicts: List[Verdict]
    
    # ── Accumulated Intelligence ──
    failure_patterns: Annotated[List[str], add]
    new_pattern_counts: Annotated[List[int], add]  # per refiner run; 0 => converged
    all_verdicts: Annotated[List[Verdict], add]
    
    # ── Metrics ──
    coverage_score: float
    iteration_pass_rates: Annotated[List[float], add]
    
    # ── Final Output ──
    final_report: Optional[str]
    usage: Optional[dict]  # token / cost summary for the run
