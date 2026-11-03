"""
LangChain Adapter — test any LangChain Runnable directly (chains, LCEL, agents).

    from langchain_classic.chains import create_retrieval_chain
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    adapter = LangChainAdapter(rag_chain, description="Research paper Q&A", input_key="input")

The answer is read from the first of ``answer`` / ``output`` / ``result`` /
``text`` (or a message's ``.content``), and retrieved documents from
``context`` / ``source_documents`` / ``documents`` — the shapes produced by
LangChain's retrieval chains — so contexts reach the grounding checks.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from agentic_qa.sut.base import BaseSUTAdapter

ANSWER_KEYS = ("answer", "output", "result", "text", "response")
CONTEXT_KEYS = ("context", "source_documents", "documents", "contexts", "retrieved_contexts")


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    content = getattr(value, "content", None)  # AIMessage / BaseMessage
    if isinstance(content, str):
        return content
    return str(value)


def _doc_text(doc: Any) -> str:
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        return str(doc.get("page_content") or doc.get("content") or doc.get("text") or "")
    return str(getattr(doc, "page_content", "") or "")


class LangChainAdapter(BaseSUTAdapter):
    def __init__(
        self,
        runnable: Any,
        description: str,
        system_name: str = "LangChain app",
        domain: str = "general",
        input_key: Optional[str] = "input",
        output_key: Optional[str] = None,
    ):
        if not hasattr(runnable, "invoke"):
            raise TypeError("LangChainAdapter needs an object with .invoke() (a LangChain Runnable)")
        self._runnable = runnable
        self._description = description
        self._name = system_name
        self._domain = domain
        self._input_key = input_key
        self._output_key = output_key

    @property
    def name(self) -> str:
        return self._name

    @property
    def domain(self) -> str:
        return self._domain

    def describe(self) -> str:
        return self._description

    def _answer(self, result: Any) -> str:
        if isinstance(result, dict):
            keys = (self._output_key,) if self._output_key else ANSWER_KEYS
            for key in keys:
                if key in result:
                    return _text(result[key])
            if self._output_key:
                raise KeyError(f"Output key '{self._output_key}' not in chain result: {list(result)}")
            return str(result)
        return _text(result)

    def process(self, input_text: str) -> dict:
        start = time.time()
        try:
            payload = {self._input_key: input_text} if self._input_key else input_text
            result = self._runnable.invoke(payload)
            contexts: list[str] = []
            if isinstance(result, dict):
                for key in CONTEXT_KEYS:
                    docs = result.get(key)
                    if isinstance(docs, (list, tuple)):
                        contexts += [t for t in (_doc_text(d) for d in docs) if t.strip()]
            return {
                "status": "success",
                "output": self._answer(result),
                "retrieved_contexts": contexts,
                "processing_time": round(time.time() - start, 4),
            }
        except Exception as e:
            return {
                "status": "error",
                "output": f"{type(e).__name__}: {e}",
                "processing_time": round(time.time() - start, 4),
            }
