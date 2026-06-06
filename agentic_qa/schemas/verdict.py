"""
Pydantic models for Judge Agent verdicts and evaluation results.

These models standardize the evaluation output from the Judge Agent,
ensuring consistent pass/fail determinations with traceable reasoning.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class VerdictStatus(str, Enum):
    """Possible outcomes of a test evaluation."""
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"       # SUT crashed or timed out
    PARTIAL = "partial"   # Partially correct output


class SeverityLevel(str, Enum):
    """Severity of a detected failure."""
    CRITICAL = "critical"   # System crash, data corruption
    HIGH = "high"           # Wrong output with significant real-world impact
    MEDIUM = "medium"       # Incorrect but non-critical extraction
    LOW = "low"             # Minor formatting or cosmetic issues
    INFO = "info"           # Observation, not a failure


class Verdict(BaseModel):
    """A single evaluation verdict from the Judge Agent."""
    test_id: str = Field(..., description="ID of the test case being evaluated")
    status: VerdictStatus = Field(..., description="Pass/fail determination")
    reasoning: str = Field(..., description="Detailed explanation of the verdict")
    severity: SeverityLevel = Field(default=SeverityLevel.MEDIUM, description="Impact severity if failed")
    failure_category: Optional[str] = Field(default=None, description="Category of failure if applicable")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Judge's confidence in this verdict")

    class Config:
        json_schema_extra = {
            "example": {
                "test_id": "TC-001",
                "status": "fail",
                "reasoning": "The SUT extracted NAV as -0.01 without flagging the negative value or the invalid date Feb 31. Expected anomaly detection was missing.",
                "severity": "high",
                "failure_category": "missing_validation",
                "confidence": 0.92
            }
        }


class VerdictBatch(BaseModel):
    """A batch of verdicts from a single Judge Agent evaluation round."""
    verdicts: list[Verdict] = Field(..., description="List of verdicts for the current iteration")
    iteration: int = Field(..., description="Which iteration these verdicts belong to")
    pass_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of tests that passed")
    summary: str = Field(default="", description="High-level summary of findings this iteration")
