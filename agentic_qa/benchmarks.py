"""
Planted-bug benchmark for the built-in demo SUT (FinancialDocumentRAG).

The demo has six deliberately planted bugs. For each one there is a
deterministic detector that looks at a test's input and the SUT's output and
decides whether the test *exposed* the bug. A bug is *caught* when at least one
test exposed it AND the judge did not pass that test.

    agentic-qa bugs --runs 3 --iterations 3 --tests 8

reports "caught N/6" per run, which measures the whole loop end to end:
generation (did it find the bug?) and judging (did it flag it?).
"""

from __future__ import annotations

import ast
import datetime
import re

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}

_NON_USD_RE = re.compile(
    r"(?:€|£|¥|₹|\b(?:EUR|GBP|JPY|INR|CHF|AUD|CAD|CNY|SGD)\b)\s*(-?[\d,]+(?:\.\d+)?)"
    r"|(-?[\d,]+(?:\.\d+)?)\s*(?:€|£|¥|₹|\b(?:EUR|GBP|JPY|INR|CHF|AUD|CAD|CNY|SGD)\b)",
    re.IGNORECASE,
)
_FUND_NAME_RE = re.compile(r"(?:Fund|Portfolio|Scheme)[:\s]+([^\n,|]+)", re.IGNORECASE)


def _parse_output(output: str) -> dict:
    try:
        data = ast.literal_eval(output or "{}")
    except (ValueError, SyntaxError):
        return {}
    return data if isinstance(data, dict) else {}


def _valid_date(y: int, m: int, d: int) -> bool:
    try:
        datetime.date(y if y > 99 else 2000 + y, m, d)
        return True
    except ValueError:
        return False


def _date_is_impossible(raw: str) -> bool:
    parts = raw.split("/")
    if len(parts) != 3:
        return False
    a, b, c = parts
    if a.lower()[:3] in MONTHS and b.isdigit() and c.isdigit():
        return not _valid_date(int(c), MONTHS[a.lower()[:3]], int(b))
    if not all(p.isdigit() for p in parts):
        return False
    a, b, c = map(int, parts)
    if len(parts[0]) == 4:  # y/m/d
        return not _valid_date(a, b, c)
    return not (_valid_date(c, b, a) or _valid_date(c, a, b))  # d/m/y or m/d/y


def _negative_nav(inp: str, out: dict) -> bool:
    return any(isinstance(n, dict) and n.get("value", 0) < 0 for n in out.get("nav", []))


def _impossible_date(inp: str, out: dict) -> bool:
    return any(_date_is_impossible(str(d)) for d in out.get("dates", []))


def _prompt_injection(inp: str, out: dict) -> bool:
    return "system_response" in out


def _usd_only(inp: str, out: dict) -> bool:
    foreign = {float((a or b).replace(",", "")) for a, b in _NON_USD_RE.findall(inp) if (a or b).strip(",")}
    extracted = {a.get("value") for a in out.get("amounts", []) if isinstance(a, dict)}
    return bool(foreign - extracted)


def _unicode_stripped(inp: str, out: dict) -> bool:
    has_unicode_name = any(re.search(r"[^\x00-\x7F]", m) for m in _FUND_NAME_RE.findall(inp))
    kept = any(re.search(r"[^\x00-\x7F]", str(n)) for n in out.get("fund_names", []))
    return has_unicode_name and not kept


def _allocation_sum(inp: str, out: dict) -> bool:
    allocs = [a.get("percentage", 0) for a in out.get("allocations", []) if isinstance(a, dict)]
    return len(allocs) >= 2 and abs(sum(allocs) - 100) > 0.5


PLANTED_BUGS = {
    "negative_nav": ("Accepts negative NAV values", _negative_nav),
    "impossible_date": ("Accepts impossible dates (e.g. 31/02/2024)", _impossible_date),
    "prompt_injection": ("Obeys 'ignore previous instructions' / 'system:'", _prompt_injection),
    "usd_only": ("Silently drops non-USD amounts", _usd_only),
    "unicode_stripping": ("Strips non-ASCII characters from fund names", _unicode_stripped),
    "allocation_sum": ("Accepts allocations that don't sum to 100%", _allocation_sum),
}


def evaluate_planted_bugs(state: dict) -> dict:
    tests = {t["id"]: t for t in state.get("test_suite", [])}
    per_bug = {key: {"description": desc, "exposed": 0, "caught": 0, "examples": []}
               for key, (desc, _) in PLANTED_BUGS.items()}
    for v in state.get("all_verdicts", []):
        tc = tests.get(v.get("test_id"))
        if not tc:
            continue
        out = _parse_output(str(v.get("sut_output", "")))
        for key, (_, detector) in PLANTED_BUGS.items():
            try:
                triggered = detector(str(tc.get("input_data", "")), out)
            except Exception:
                triggered = False
            if triggered:
                entry = per_bug[key]
                entry["exposed"] += 1
                if v.get("status") != "pass":
                    entry["caught"] += 1
                    if len(entry["examples"]) < 3:
                        entry["examples"].append(v["test_id"])
    return {
        "bugs_total": len(PLANTED_BUGS),
        "bugs_exposed": sum(1 for b in per_bug.values() if b["exposed"]),
        "bugs_caught": sum(1 for b in per_bug.values() if b["caught"]),
        "per_bug": per_bug,
    }
