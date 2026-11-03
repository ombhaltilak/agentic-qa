"""
Compare two runs — "did my change make things better or worse, and where?"

Typical workflow after changing a prompt, model, chunk size or retriever:

    agentic-qa run  --target my_rag:answer --dataset golden.jsonl --output baseline.json
    # ... change something ...
    agentic-qa run  --target my_rag:answer --replay baseline.json --output candidate.json
    agentic-qa diff baseline.json candidate.json --markdown diff.md --fail-on-new-failures

Tests are matched by their (normalized) input text, so a replay of the same
suite lines up exactly; tests present in only one run are listed separately.
Pass rates are compared on the shared tests only (apples to apples).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from agentic_qa.taxonomy import normalize_category


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def load_run(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or "verdicts" not in data or "test_suite" not in data:
        raise ValueError(f"{path} is not a run file (expected 'test_suite' and 'verdicts')")
    return data


def _index(run: dict) -> dict[str, dict]:
    verdicts = {v.get("test_id"): v for v in run.get("verdicts", [])}
    index: dict[str, dict] = {}
    for tc in run.get("test_suite", []):
        v = verdicts.get(tc.get("id"))
        k = _key(tc.get("input_data"))
        if v is None or k in index or v.get("status") == "invalid":  # bad tests aren't compared
            continue
        index[k] = {"test": tc, "verdict": v, "passed": v.get("status") == "pass"}
    return index


def _metric(run: dict, *path):
    cur = run.get("summary", {})
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _item(entry_a: dict | None, entry_b: dict | None) -> dict:
    ref = entry_b or entry_a
    tc = ref["test"]
    item = {
        "input": str(tc.get("input_data", ""))[:200],
        "category": normalize_category(tc.get("edge_case_type")),
    }
    if entry_a:
        item["before"] = {"status": entry_a["verdict"].get("status"),
                          "output": str(entry_a["verdict"].get("sut_output", ""))[:300]}
    if entry_b:
        item["after"] = {"status": entry_b["verdict"].get("status"),
                         "output": str(entry_b["verdict"].get("sut_output", ""))[:300],
                         "reason": str(entry_b["verdict"].get("reasoning", ""))[:300],
                         "severity": entry_b["verdict"].get("severity")}
    return item


def compare_runs(run_a: dict, run_b: dict) -> dict:
    a, b = _index(run_a), _index(run_b)
    common = sorted(set(a) & set(b))

    newly_failing = [_item(a[k], b[k]) for k in common if a[k]["passed"] and not b[k]["passed"]]
    fixed = [_item(a[k], b[k]) for k in common if not a[k]["passed"] and b[k]["passed"]]
    still_failing = [_item(a[k], b[k]) for k in common if not a[k]["passed"] and not b[k]["passed"]]

    def rate(index, keys):
        return round(sum(index[k]["passed"] for k in keys) / len(keys), 4) if keys else None

    per_cat = defaultdict(lambda: {"tests": 0, "before": 0, "after": 0})
    for k in common:
        cat = normalize_category(a[k]["test"].get("edge_case_type"))
        per_cat[cat]["tests"] += 1
        per_cat[cat]["before"] += a[k]["passed"]
        per_cat[cat]["after"] += b[k]["passed"]
    categories = {
        c: {"tests": s["tests"], "pass_rate_before": round(s["before"] / s["tests"], 4),
            "pass_rate_after": round(s["after"] / s["tests"], 4)}
        for c, s in sorted(per_cat.items())
    }

    metrics = {}
    for name, path, better in [
        ("hallucination_rate", ("golden", "hallucination_rate"), "lower"),
        ("critical_or_high_failures", ("critical_or_high_failures",), "lower"),
        ("policy_violation_tests", ("policy_violation_tests",), "lower"),
        ("latency_p50_s", ("sut_latency", "p50_s"), "lower"),
        ("latency_p95_s", ("sut_latency", "p95_s"), "lower"),
        ("qa_cost_usd", ("usage", "estimated_cost_usd"), "lower"),
        ("coverage", ("coverage",), "higher"),
    ]:
        before, after = _metric(run_a, *path), _metric(run_b, *path)
        if before is None and after is None:
            continue
        delta = round(after - before, 4) if isinstance(before, (int, float)) and isinstance(after, (int, float)) else None
        metrics[name] = {"before": before, "after": after, "delta": delta, "better": better}

    return {
        "common_tests": len(common),
        "only_in_before": len(set(a) - set(b)),
        "only_in_after": len(set(b) - set(a)),
        "pass_rate_before": rate(a, common),
        "pass_rate_after": rate(b, common),
        "newly_failing": newly_failing,
        "fixed": fixed,
        "still_failing": still_failing,
        "categories": categories,
        "metrics": metrics,
        "verdict": "regressed" if newly_failing else ("improved" if fixed else "unchanged"),
    }


def _fmt(v, pct=False):
    if v is None:
        return "—"
    if pct:
        return f"{v:.1%}"
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def _trend(m: dict) -> str:
    d = m["delta"]
    if d is None or d == 0:
        return ""
    good = (d < 0) if m["better"] == "lower" else (d > 0)
    return "✅" if good else "❌"


def to_markdown(diff: dict, name_a: str = "before", name_b: str = "after") -> str:
    icon = {"regressed": "❌ Regressed", "improved": "✅ Improved", "unchanged": "➖ Unchanged"}[diff["verdict"]]
    lines = [
        f"## Agentic QA diff: {icon}",
        "",
        f"`{name_a}` → `{name_b}` · {diff['common_tests']} shared tests"
        + (f" · {diff['only_in_before']} only before · {diff['only_in_after']} only after"
           if diff["only_in_before"] or diff["only_in_after"] else ""),
        "",
        "| Metric | Before | After | |",
        "|---|---|---|---|",
        f"| Pass rate (shared tests) | {_fmt(diff['pass_rate_before'], True)} | {_fmt(diff['pass_rate_after'], True)} | "
        + ("✅" if (diff['pass_rate_after'] or 0) > (diff['pass_rate_before'] or 0) else
           "❌" if (diff['pass_rate_after'] or 0) < (diff['pass_rate_before'] or 0) else "") + " |",
    ]
    for name, m in diff["metrics"].items():
        pct = name in ("hallucination_rate", "coverage")
        lines.append(f"| {name.replace('_', ' ')} | {_fmt(m['before'], pct)} | {_fmt(m['after'], pct)} | {_trend(m)} |")

    def section(title, items, limit=15):
        if not items:
            return
        lines.extend(["", f"### {title} ({len(items)})", ""])
        for it in items[:limit]:
            after = it.get("after", {})
            lines.append(f"- **[{it['category']}]** {it['input']}")
            if after.get("reason"):
                lines.append(f"  - now: `{after.get('status')}` — {after['reason'][:200]}")
        if len(items) > limit:
            lines.append(f"- … and {len(items) - limit} more")

    section("❌ Newly failing", diff["newly_failing"])
    section("✅ Fixed", diff["fixed"])
    changed = {c: s for c, s in diff["categories"].items() if s["pass_rate_before"] != s["pass_rate_after"]}
    if changed:
        lines.extend(["", "### Categories that changed", "", "| Category | Tests | Before | After |", "|---|---|---|---|"])
        for c, s in changed.items():
            lines.append(f"| {c} | {s['tests']} | {s['pass_rate_before']:.0%} | {s['pass_rate_after']:.0%} |")
    lines.extend(["", f"_{len(diff['still_failing'])} test(s) failing in both runs._"])
    return "\n".join(lines)
