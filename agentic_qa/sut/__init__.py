"""Multi-Agent Autonomous QA System - SUT Package

Provides adapter classes to connect ANY system for testing:
  - FinancialDocumentRAG: Built-in demo SUT
  - APIAdapter: Connect any RAG via HTTP API endpoint
  - CallableAdapter: Wrap any Python function
"""

from agentic_qa.sut.base import BaseSUTAdapter
from agentic_qa.sut.api_adapter import APIAdapter
from agentic_qa.sut.callable_adapter import CallableAdapter
from agentic_qa.sut.financial_rag import FinancialDocumentRAG

# Global SUT instance — set by the user or defaults to built-in demo
_active_sut: BaseSUTAdapter = None


def set_active_sut(sut: BaseSUTAdapter):
    """Set the active System Under Test."""
    global _active_sut
    _active_sut = sut


def get_active_sut() -> BaseSUTAdapter:
    """Get the active SUT. Falls back to built-in demo if none set."""
    global _active_sut
    if _active_sut is None:
        _active_sut = FinancialDocumentRAG()
    return _active_sut


__all__ = [
    "BaseSUTAdapter",
    "APIAdapter",
    "CallableAdapter",
    "FinancialDocumentRAG",
    "set_active_sut",
    "get_active_sut",
]
