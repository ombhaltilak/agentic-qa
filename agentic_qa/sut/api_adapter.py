"""
API Adapter — Connect ANY RAG system via its HTTP API endpoint.

Works with any REST API (FastAPI, Flask, LangServe, ...). Real APIs are rarely
flat, so the adapter supports:

  - nested response paths:   output_path="data.answer", contexts_path="data.sources.*.text"
  - request body templates:  body_template={"messages": [{"role": "user", "content": "{input}"}]}
  - secret headers from env: headers={"Authorization": "Bearer ${MY_API_KEY}"}

``OpenAIChatAdapter`` is a ready-made subclass for OpenAI-compatible
``/v1/chat/completions`` endpoints (OpenAI, vLLM, Ollama, LM Studio, OpenRouter...).
"""

from __future__ import annotations

import copy
import os
import re
import time
from typing import Any, Optional

import requests

from agentic_qa.sut.base import BaseSUTAdapter

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_MISSING = object()


def expand_env(value: str) -> str:
    """Replace ${VAR} with the environment value. Raises if a variable is unset."""
    def repl(m):
        name = m.group(1)
        if name not in os.environ:
            raise ValueError(f"Environment variable {name} is not set (used in a header)")
        return os.environ[name]
    return _ENV_RE.sub(repl, value)


def get_path(data: Any, path: Optional[str], default: Any = _MISSING) -> Any:
    """
    Read a dotted path: "data.answer", "choices.0.message.content".
    "*" maps over a list: "sources.*.text" -> [s["text"] for s in sources].
    """
    if not path:
        return data
    parts = path.split(".")
    current = data
    for i, part in enumerate(parts):
        if part == "*":
            if not isinstance(current, list):
                return default if default is not _MISSING else None
            rest = ".".join(parts[i + 1:])
            return [get_path(item, rest, None) for item in current]
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.lstrip("-").isdigit() and -len(current) <= int(part) < len(current):
            current = current[int(part)]
        else:
            if default is not _MISSING:
                return default
            raise KeyError(f"Path '{path}' not found in response (failed at '{part}')")
    return current


def fill_template(template: Any, text: str) -> Any:
    """Deep-copy a JSON template, substituting "{input}" inside string values (JSON-safe)."""
    if isinstance(template, str):
        return template.replace("{input}", text)
    if isinstance(template, list):
        return [fill_template(t, text) for t in template]
    if isinstance(template, dict):
        return {k: fill_template(v, text) for k, v in template.items()}
    return copy.deepcopy(template)


class APIAdapter(BaseSUTAdapter):
    """
    Connect to any RAG system via HTTP API.

    Usage:
        adapter = APIAdapter(
            endpoint="http://localhost:8000/chat",
            description="YouTube video Q&A chatbot",
            input_key="question",          # flat body: {"question": ...}
            output_path="data.answer",     # nested response field
            contexts_path="data.sources.*.text",
            headers={"Authorization": "Bearer ${CHATBOT_API_KEY}"},
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
        output_path: Optional[str] = None,
        contexts_path: Optional[str] = None,
        body_template: Optional[Any] = None,
    ):
        self._endpoint = endpoint
        self._description = description
        self._name = system_name
        self._domain = domain
        self._input_key = input_key
        self._output_path = output_path or output_key
        self._contexts_path = contexts_path or contexts_key
        self._sources_key = sources_key
        self._metadata_key = metadata_key
        self._headers = {"Content-Type": "application/json", **(headers or {})}
        self._timeout = timeout
        self._method = method.upper()
        self._body_template = body_template
        # Fail fast on missing env vars rather than on every request.
        for value in self._headers.values():
            expand_env(str(value))

    @property
    def name(self) -> str:
        return self._name

    @property
    def domain(self) -> str:
        return self._domain

    def describe(self) -> str:
        return self._description

    def _request(self, input_text: str) -> requests.Response:
        headers = {k: expand_env(str(v)) for k, v in self._headers.items()}
        if self._method == "GET":
            return requests.get(self._endpoint, params={self._input_key: input_text},
                                headers=headers, timeout=self._timeout)
        body = fill_template(self._body_template, input_text) if self._body_template is not None \
            else {self._input_key: input_text}
        return requests.request(self._method, self._endpoint, json=body, headers=headers, timeout=self._timeout)

    def process(self, input_text: str) -> dict:
        """Send input to the API endpoint and return the normalized response."""
        start_time = time.time()
        try:
            response = self._request(input_text)
            elapsed = time.time() - start_time

            if not 200 <= response.status_code < 300:
                return {
                    "status": "error",
                    "output": f"HTTP {response.status_code}: {response.text[:500]}",
                    "processing_time": round(elapsed, 4),
                }
            try:
                data = response.json()
            except ValueError:
                return {"status": "success", "output": response.text,
                        "processing_time": round(elapsed, 4)}

            try:
                output = get_path(data, self._output_path)
            except KeyError as e:
                return {"status": "error", "output": f"{e}. Response keys: {list(data)[:10] if isinstance(data, dict) else type(data).__name__}",
                        "processing_time": round(elapsed, 4)}

            contexts = get_path(data, self._contexts_path, []) if self._contexts_path else []
            if not isinstance(contexts, list):
                contexts = [contexts] if contexts else []
            return {
                "status": "success",
                "output": output if isinstance(output, str) else str(output),
                "retrieved_contexts": [c for c in contexts if c],
                "sources": get_path(data, self._sources_key, []) if self._sources_key else [],
                "metadata": get_path(data, self._metadata_key, {}) if self._metadata_key else {},
                "full_response": data,
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
                "output": f"{type(e).__name__}: {e}",
                "processing_time": round(time.time() - start_time, 4),
            }


class OpenAIChatAdapter(APIAdapter):
    """
    Test any OpenAI-compatible chat endpoint (OpenAI, vLLM, Ollama, LM Studio, OpenRouter...).

        OpenAIChatAdapter(base_url="http://localhost:11434/v1", model="llama3.1",
                          description="Support bot", system_prompt="You are ...")
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        description: str,
        system_name: str = "Chat model",
        domain: str = "general",
        api_key_env: Optional[str] = "OPENAI_API_KEY",
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        timeout: int = 60,
    ):
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + \
                   [{"role": "user", "content": "{input}"}]
        headers = {"Authorization": f"Bearer ${{{api_key_env}}}"} if api_key_env else {}
        super().__init__(
            endpoint=base_url.rstrip("/") + "/chat/completions",
            description=description,
            system_name=system_name,
            domain=domain,
            output_path="choices.0.message.content",
            contexts_key=None,
            sources_key=None,
            metadata_key=None,
            headers=headers,
            timeout=timeout,
            body_template={"model": model, "messages": messages, "temperature": temperature},
        )
