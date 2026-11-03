"""Verdict helpers shared by the judge, summaries, diffs and memory.

Besides pass / fail / partial / error, a verdict can be ``invalid``: the test itself was
wrong (its expected behavior contradicts the system's purpose, policy or sources) and the
output was fine on its merits. Invalid tests are excluded from pass rates, failure counts,
gates and diffs, never stored as regressions, and shown to the Red-Team as what not to write.
"""

from __future__ import annotations

INVALID = "invalid"


def is_invalid(verdict: dict) -> bool:
    return verdict.get("status") == INVALID


def scored(verdicts: list[dict]) -> list[dict]:
    """Verdicts that count toward quality metrics."""
    return [v for v in verdicts or [] if not is_invalid(v)]
