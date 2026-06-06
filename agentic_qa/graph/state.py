"""
LangGraph State Schema for the Multi-Agent QA System.

Defines the shared state (QAState) that flows between all agent nodes
in the LangGraph workflow. Each agent reads from and writes to this
shared state, enabling fully autonomous multi-agent coordination.
"""

from typing import TypedDict, List, Optional, Annotated
from operator import add


class TestCase(TypedDict):
    """A single test case in the pipeline state."""
    id: str
    input_data: str
    expected_behavior: str
    edge_case_type: str
    difficulty: str
    rationale: str


class TestResult(TypedDict):
    """Result from executing a test case against the SUT."""
    test_id: str
    sut_output: str
    execution_time: float
    error: Optional[str]


class Verdict(TypedDict):
    """Judge Agent's evaluation of a test result."""
    test_id: str
    status: str  # "pass", "fail", "error", "partial"
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
    sut_description: str
    sut_architecture: Optional[str]
    domain: str  # e.g., "academic", "customer support", "legal", "general"
    benchmark_mode: Optional[bool]
    
    
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
    all_verdicts: Annotated[List[Verdict], add]
    
    # ── Metrics ──
    coverage_score: float
    iteration_pass_rates: Annotated[List[float], add]
    
    # ── Final Output ──
    final_report: Optional[str]
