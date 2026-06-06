"""
API Adapter — Connect ANY RAG system via its HTTP API endpoint.

The user just provides a URL and the adapter handles the rest.
Works with any RAG that has a REST API (FastAPI, Flask, LangServe, etc.)
"""

import time
import requests
from typing import Optional
from agentic_qa.sut.base import BaseSUTAdapter


class APIAdapter(BaseSUTAdapter):
    """
    Connect to any RAG system via HTTP API.
    
    Usage:
        adapter = APIAdapter(
            endpoint="http://localhost:8000/chat",
            description="YouTube video Q&A chatbot",
            domain="education",
            input_key="question",     # JSON key for the input
            output_key="answer",      # JSON key in the response
        )
    """
    
    def __init__(
        self,
        endpoint: str,
        description: str,
        system_name: str = "User RAG System",
        domain: str = "general",
        input_key: str = "query",
        output_key: str = "output",
        contexts_key: Optional[str] = "contexts",
        sources_key: Optional[str] = "sources",
        metadata_key: Optional[str] = "metadata",
        headers: Optional[dict] = None,
        timeout: int = 30,
        method: str = "POST",
    ):
        self._endpoint = endpoint
        self._description = description
        self._name = system_name
        self._domain = domain
        self._input_key = input_key
        self._output_key = output_key
        self._contexts_key = contexts_key
        self._sources_key = sources_key
        self._metadata_key = metadata_key
        self._headers = headers or {"Content-Type": "application/json"}
        self._timeout = timeout
        self._method = method.upper()
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def domain(self) -> str:
        return self._domain
    
    def describe(self) -> str:
        return self._description
    
    def process(self, input_text: str) -> dict:
        """Send input to the API endpoint and return the response."""
        start_time = time.time()
        
        try:
            if self._method == "POST":
                payload = {self._input_key: input_text}
                response = requests.post(
                    self._endpoint,
                    json=payload,
                    headers=self._headers,
                    timeout=self._timeout,
                )
            else:
                params = {self._input_key: input_text}
                response = requests.get(
                    self._endpoint,
                    params=params,
                    headers=self._headers,
                    timeout=self._timeout,
                )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                # Try to extract the output using the configured key
                output = data.get(self._output_key, str(data))
                
                retrieved_contexts = data.get(self._contexts_key, []) if self._contexts_key else []
                sources = data.get(self._sources_key, []) if self._sources_key else []
                metadata = data.get(self._metadata_key, {}) if self._metadata_key else {}
                
                return {
                    "status": "success",
                    "output": str(output),
                    "retrieved_contexts": retrieved_contexts,
                    "sources": sources,
                    "metadata": metadata,
                    "full_response": data,
                    "processing_time": round(elapsed, 4),
                }
            else:
                return {
                    "status": "error",
                    "output": f"HTTP {response.status_code}: {response.text[:500]}",
                    "processing_time": round(elapsed, 4),
                }
                
        except requests.exceptions.ConnectionError:
            return {
                "status": "error",
                "output": f"Cannot connect to {self._endpoint}. Is the server running?",
                "processing_time": round(time.time() - start_time, 4),
            }
        except requests.exceptions.Timeout:
            return {
                "status": "error",
                "output": f"Request timed out after {self._timeout}s",
                "processing_time": round(time.time() - start_time, 4),
            }
        except Exception as e:
            return {
                "status": "error",
                "output": str(e),
                "processing_time": round(time.time() - start_time, 4),
            }
