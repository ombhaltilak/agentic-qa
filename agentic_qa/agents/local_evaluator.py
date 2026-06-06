"""
Local evaluation helpers for the Judge Agent.

These checks are intentionally dependency-free so the QA pipeline still gains
useful signals when paid/API-based metrics such as RAGAS are unavailable.
"""

from __future__ import annotations

import re
import ast
from difflib import SequenceMatcher
from html import unescape


_NEGATIVE_PATTERNS = (
    r"\bno output\b",
    r"\bi don'?t know\b",
    r"\bnot sure\b",
    r"\bcannot answer\b",
    r"\bunable to\b",
    r"\berror\b",
    r"\bexception\b",
    r"\btraceback\b",
)

_INJECTION_PATTERNS = (
    r"<\s*script\b",
    r"javascript\s*:",
    r"on\w+\s*=",
    r"ignore (all )?(previous|prior) instructions",
    r"system prompt",
    r"developer message",
    r"drop\s+table",
    r"union\s+select",
    r"\{\{.*\}\}",
)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "that", "the", "this", "to", "with", "should", "must",
    "will", "would", "could", "input", "output", "system", "sut", "data", "value",
}

_CONTEXT_KEYS = (
    "contexts",
    "retrieved_contexts",
    "context",
    "source_documents",
    "documents",
    "sources",
)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", text.lower())
        if token not in _STOPWORDS
    }


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _contains_any(text: str, patterns: tuple[str, ...]) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)]


def _as_dict(text: str) -> dict:
    try:
        data = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return {}
    return data if isinstance(data, dict) else {}


def _flatten_contexts(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        for key in ("page_content", "content", "text", "body", "snippet"):
            if key in value:
                return _flatten_contexts(value[key])
        return [str(value)] if value else []
    if isinstance(value, (list, tuple)):
        flattened: list[str] = []
        for item in value:
            flattened.extend(_flatten_contexts(item))
        return flattened
    return [str(value)]


def _extract_contexts(raw_output: str) -> list[str]:
    parsed = _as_dict(raw_output)
    contexts: list[str] = []
    for key in _CONTEXT_KEYS:
        if key in parsed:
            contexts.extend(_flatten_contexts(parsed[key]))
    return [ctx.strip() for ctx in contexts if ctx and ctx.strip()]


def _sentence_claims(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [
        sentence.strip()
        for sentence in sentences
        if len(_tokens(sentence)) >= 3 and not sentence.strip().startswith("{")
    ][:12]


def _overlap_ratio(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    return len(left_tokens & right_tokens) / max(len(left_tokens), 1)


def evaluate_result(result: dict) -> dict:
    """Return local quality and risk signals for one combined test result."""
    input_data = str(result.get("input_data", ""))
    expected = str(result.get("expected_behavior", ""))
    output = str(result.get("sut_output", ""))
    error = result.get("execution_error")

    normalized_output = unescape(output).strip()
    contexts = _extract_contexts(normalized_output)
    context_text = "\n".join(contexts)
    expected_tokens = _tokens(expected)
    output_tokens = _tokens(normalized_output)
    input_tokens = _tokens(input_data)

    expected_overlap = len(expected_tokens & output_tokens) / max(len(expected_tokens), 1)
    input_overlap = len(input_tokens & output_tokens) / max(len(input_tokens), 1)
    lexical_similarity = SequenceMatcher(None, expected.lower(), normalized_output.lower()).ratio()

    negative_hits = _contains_any(normalized_output, _NEGATIVE_PATTERNS)
    injection_hits = _contains_any(input_data + "\n" + normalized_output, _INJECTION_PATTERNS)
    empty_output = not normalized_output or normalized_output == "NO OUTPUT"
    crashed = bool(error)
    context_relevance = _overlap_ratio(input_data + " " + expected, context_text) if contexts else 0.0
    context_answer_support = _overlap_ratio(normalized_output, context_text) if contexts else 0.0
    context_noise = 1.0 - _clamp(context_relevance) if contexts else 1.0
    missed_expected_facts = sorted(expected_tokens - output_tokens)[:12]

    unsupported_claims = []
    contradicted_claims = []
    if contexts:
        context_lower = context_text.lower()
        for claim in _sentence_claims(normalized_output):
            support = _overlap_ratio(claim, context_text)
            if support < 0.22:
                unsupported_claims.append(claim[:220])
            if re.search(r"\bnot\b|\bno\b|\bnever\b", claim.lower()):
                positive = re.sub(r"\b(not|no|never)\b", "", claim.lower()).strip()
                if positive and positive in context_lower:
                    contradicted_claims.append(claim[:220])

    if not contexts:
        grounding_verdict = "not_enough_context"
    elif contradicted_claims:
        grounding_verdict = "contradicted"
    elif unsupported_claims:
        grounding_verdict = "unsupported"
    else:
        grounding_verdict = "grounded"

    length_score = _clamp(len(normalized_output) / 280) if normalized_output else 0.0
    quality_score = _clamp(
        (expected_overlap * 0.42)
        + (lexical_similarity * 0.22)
        + (min(input_overlap, 0.6) * 0.16)
        + (length_score * 0.20)
        - (0.25 if negative_hits else 0.0)
        - (0.35 if empty_output else 0.0)
        - (0.45 if crashed else 0.0)
    )

    risk_score = _clamp(
        (0.35 if injection_hits else 0.0)
        + (0.30 if negative_hits else 0.0)
        + (0.45 if empty_output else 0.0)
        + (0.55 if crashed else 0.0)
        + (0.25 if grounding_verdict in {"unsupported", "contradicted"} else 0.0)
    )

    if quality_score >= 0.74 and risk_score < 0.35:
        recommendation = "strong_local_signal"
    elif risk_score >= 0.55:
        recommendation = "high_risk_review"
    elif quality_score < 0.38:
        recommendation = "weak_answer_review"
    else:
        recommendation = "needs_judge_review"

    return {
        "local_quality_score": round(quality_score, 4),
        "expected_keyword_coverage": round(expected_overlap, 4),
        "input_keyword_coverage": round(input_overlap, 4),
        "lexical_similarity": round(lexical_similarity, 4),
        "risk_score": round(risk_score, 4),
        "retrieval_inspection": {
            "contexts_found": len(contexts),
            "context_relevance": round(context_relevance, 4),
            "context_answer_support": round(context_answer_support, 4),
            "context_noise": round(context_noise, 4),
            "missed_expected_facts": missed_expected_facts,
        },
        "hallucination_detection": {
            "verdict": grounding_verdict,
            "unsupported_claims": unsupported_claims[:5],
            "contradicted_claims": contradicted_claims[:5],
        },
        "security_flags": injection_hits,
        "negative_response_flags": negative_hits,
        "empty_output": empty_output,
        "execution_error": crashed,
        "recommendation": recommendation,
    }


def evaluate_batch(combined_results: list[dict]) -> dict[str, dict]:
    """Evaluate a batch and return scores keyed by test id."""
    return {
        str(result.get("test_id", "unknown")): evaluate_result(result)
        for result in combined_results
    }
