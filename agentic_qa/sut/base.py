"""
Base SUT Adapter — Abstract interface for any System Under Test.

Any RAG system can be tested by implementing this simple interface.
The user just needs to define how to send input and get output.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseSUTAdapter(ABC):
    """
    Base class for connecting ANY system to the QA pipeline.
    
    To test your own RAG, extend this class and implement:
      - process(input_text) -> dict
      - describe() -> str
    
    That's it. The agents handle everything else.
    """
    
    @abstractmethod
    def process(self, input_text: str) -> dict:
        """
        Send input to the system and return its response.
        
        Args:
            input_text: The test input (query, document, prompt, etc.)
        Returns:
            Dict with at least:
                - "status": "success" | "error"
                - "output": The system's response (string)
                - "retrieved_contexts": Optional[list[str]] (Optional)
                - "sources": Optional[list[str]] (Optional)
                - "metadata": Optional[dict] (Optional)
        """
        pass
    
    @abstractmethod
    def describe(self) -> str:
        """
        Return a human-readable description of what this system does.
        This is used by the agents to understand what they're testing.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the system being tested."""
        pass
    
    @property
    def domain(self) -> str:
        """Domain of the system (e.g., 'financial', 'medical', 'general')."""
        return "general"
