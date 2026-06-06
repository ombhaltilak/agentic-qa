"""
Callable Adapter — Wrap any Python function as a SUT.

For users who want to test a RAG that exists as Python code
in the same environment (not behind an API).
"""

import time
from typing import Callable
from agentic_qa.sut.base import BaseSUTAdapter


class CallableAdapter(BaseSUTAdapter):
    """
    Wrap any Python function/callable as a testable SUT.
    
    Usage:
        def my_rag(query: str) -> str:
            return my_chain.invoke({"question": query})
        
        adapter = CallableAdapter(
            fn=my_rag,
            description="YouTube video Q&A chatbot using ChromaDB",
            domain="education",
        )
    """
    
    def __init__(
        self,
        fn: Callable,
        description: str,
        system_name: str = "User RAG System",
        domain: str = "general",
    ):
        self._fn = fn
        self._description = description
        self._name = system_name
        self._domain = domain
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def domain(self) -> str:
        return self._domain
    
    def describe(self) -> str:
        return self._description
    
    def process(self, input_text: str) -> dict:
        """Call the wrapped function and return the result."""
        start_time = time.time()
        
        try:
            result = self._fn(input_text)
            elapsed = time.time() - start_time
            
            # Handle different return types
            if isinstance(result, dict):
                output = result.get("output", result.get("answer", result.get("response", str(result))))
                payload = {
                    "status": "success",
                    "output": str(output),
                    "full_response": result,
                    "processing_time": round(elapsed, 4),
                }
                for key in ("contexts", "retrieved_contexts", "context", "source_documents", "documents", "sources"):
                    if key in result:
                        payload[key] = result[key]
                return payload
            else:
                return {
                    "status": "success",
                    "output": str(result),
                    "processing_time": round(elapsed, 4),
                }
                
        except Exception as e:
            return {
                "status": "error",
                "output": str(e),
                "processing_time": round(time.time() - start_time, 4),
            }
