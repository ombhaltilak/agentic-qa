"""
Reporter Agent — Final QA Report Generator.

Compiles a comprehensive, professional QA report from all iterations
of the multi-agent testing session.
"""

import json
import os
from collections import Counter
from html import escape
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from agentic_qa.graph.state import QAState
from agentic_qa.utils.prompt_templates import REPORTER_SYSTEM_PROMPT, REPORTER_GENERATION_PROMPT


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        temperature=0.2,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
    )


def _label(value: object, default: str = "unknown") -> str:
    if value is None:
        return default
    return str(value).strip().lower() or default


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def _report_metrics(state: QAState) -> dict:
    verdicts = state.get("all_verdicts", []) or state.get("judge_verdicts", [])
    tests = state.get("test_suite", [])
    pass_rates = state.get("iteration_pass_rates", [])
    statuses = Counter(_label(v.get("status")) for v in verdicts)
    severities = Counter(_label(v.get("severity"), "info") for v in verdicts)
    categories = Counter(
        _label(v.get("failure_category"), "uncategorized")
        for v in verdicts
        if _label(v.get("status")) != "pass"
    )
    edge_cases = Counter(_label(t.get("edge_case_type")) for t in tests)
    total = sum(statuses.values())
    passing = statuses.get("pass", 0)
    current_pass = pass_rates[-1] if pass_rates else (passing / total if total else 0.0)
    first_pass = pass_rates[0] if pass_rates else current_pass
    confidence = [float(v.get("confidence", 0.0)) for v in verdicts if isinstance(v.get("confidence", 0.0), (int, float))]
    local_evals = [v.get("local_evaluation", {}) for v in verdicts if v.get("local_evaluation")]
    local_quality = [float(e.get("local_quality_score", 0.0)) for e in local_evals if isinstance(e.get("local_quality_score", 0.0), (int, float))]
    local_risk = [float(e.get("risk_score", 0.0)) for e in local_evals if isinstance(e.get("risk_score", 0.0), (int, float))]
    local_recommendations = Counter(_label(e.get("recommendation"), "unknown") for e in local_evals)
    retrieval_evals = [e.get("retrieval_inspection", {}) for e in local_evals]
    hallucination_evals = [e.get("hallucination_detection", {}) for e in local_evals]
    context_counts = [int(e.get("contexts_found", 0)) for e in retrieval_evals if isinstance(e.get("contexts_found", 0), int)]
    context_relevance = [float(e.get("context_relevance", 0.0)) for e in retrieval_evals if isinstance(e.get("context_relevance", 0.0), (int, float))]
    grounding = Counter(_label(e.get("verdict"), "unknown") for e in hallucination_evals)
    security_alerts = sum(1 for e in local_evals if e.get("security_flags"))
    return {
        "tests": len(tests),
        "verdicts": total,
        "failures": total - passing,
        "current_pass": current_pass,
        "best_pass": max(pass_rates) if pass_rates else current_pass,
        "delta": current_pass - first_pass,
        "coverage": float(state.get("coverage_score", 0.0) or 0.0),
        "confidence": sum(confidence) / len(confidence) if confidence else 0.0,
        "local_quality": sum(local_quality) / len(local_quality) if local_quality else 0.0,
        "local_risk": sum(local_risk) / len(local_risk) if local_risk else 0.0,
        "security_alerts": security_alerts,
        "avg_contexts": sum(context_counts) / len(context_counts) if context_counts else 0.0,
        "context_relevance": sum(context_relevance) / len(context_relevance) if context_relevance else 0.0,
        "grounding": grounding,
        "local_recommendations": local_recommendations,
        "pass_rates": pass_rates,
        "statuses": statuses,
        "severities": severities,
        "categories": categories,
        "edge_cases": edge_cases,
        "patterns": state.get("failure_patterns", []) or [],
    }


def _kpi(label: str, value: str, detail: str, tone: str) -> str:
    return f'''
        <article class="kpi kpi-{tone}">
            <span>{escape(label)}</span>
            <strong>{escape(value)}</strong>
            <small>{escape(detail)}</small>
        </article>
    '''


def _bar_chart(title: str, counts: Counter, colors: list[str], empty: str) -> str:
    if not counts:
        return f'''<section class="panel chart-panel"><div class="panel-heading"><h2>{escape(title)}</h2></div><p class="empty-chart">{escape(empty)}</p></section>'''
    max_count = max(counts.values()) or 1
    rows = []
    for index, (label, count) in enumerate(counts.most_common(8)):
        width = (count / max_count) * 100
        rows.append(f'''
            <div class="bar-row">
                <div class="bar-meta"><span>{escape(label.replace("_", " ").title())}</span><strong>{count}</strong></div>
                <div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%;background:{colors[index % len(colors)]};"></div></div>
            </div>
        ''')
    return f'''<section class="panel chart-panel"><div class="panel-heading"><h2>{escape(title)}</h2></div><div class="bar-chart">{''.join(rows)}</div></section>'''


def _donut_chart(title: str, counts: Counter) -> str:
    total = sum(counts.values())
    colors = {"pass": "#21a67a", "fail": "#ef476f", "error": "#ff9f1c", "partial": "#3a86ff", "unknown": "#7c8a9a"}
    if not total:
        segments = '<circle r="64" cx="90" cy="90" fill="none" stroke="#263244" stroke-width="26" />'
        legend = '<li><span style="background:#7c8a9a"></span>No verdicts yet</li>'
    else:
        circumference = 2 * 3.14159 * 64
        offset = 0
        parts = []
        items = []
        for label, count in counts.most_common():
            color = colors.get(label, "#7c8a9a")
            dash = (count / total) * circumference
            parts.append(f'<circle r="64" cx="90" cy="90" fill="none" stroke="{color}" stroke-width="26" stroke-dasharray="{dash:.2f} {circumference - dash:.2f}" stroke-dashoffset="-{offset:.2f}" />')
            items.append(f'<li><span style="background:{color}"></span>{escape(label.title())}: <strong>{count}</strong></li>')
            offset += dash
        segments = "".join(parts)
        legend = "".join(items)
    return f'''
        <section class="panel donut-panel">
            <div class="panel-heading"><h2>{escape(title)}</h2></div>
            <div class="donut-layout">
                <svg viewBox="0 0 180 180" role="img" aria-label="{escape(title)} chart">
                    <circle r="64" cx="90" cy="90" fill="none" stroke="#263244" stroke-width="26" />
                    <g transform="rotate(-90 90 90)">{segments}</g>
                    <text x="90" y="86" text-anchor="middle" class="donut-value">{total}</text>
                    <text x="90" y="106" text-anchor="middle" class="donut-label">verdicts</text>
                </svg>
                <ul class="legend">{legend}</ul>
            </div>
        </section>
    '''


def _line_chart(pass_rates: list[float]) -> str:
    if not pass_rates:
        return '''<section class="panel trend-panel"><div class="panel-heading"><h2>Pass Rate Trend</h2></div><p class="empty-chart">No iteration pass-rate data available.</p></section>'''
    width, height, pad = 640, 240, 34
    if len(pass_rates) == 1:
        points = [(pad, height - pad - (pass_rates[0] * (height - pad * 2)))]
    else:
        step = (width - pad * 2) / (len(pass_rates) - 1)
        points = [(pad + i * step, height - pad - rate * (height - pad * 2)) for i, rate in enumerate(pass_rates)]
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    area = f"{pad},{height - pad} {polyline} {width - pad},{height - pad}"
    markers = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5"><title>Iteration {i + 1}: {_pct(pass_rates[i])}</title></circle>' for i, (x, y) in enumerate(points))
    labels = "".join(f'<span style="left:{(x / width) * 100:.1f}%;">{i + 1}</span>' for i, (x, _) in enumerate(points))
    return f'''
        <section class="panel trend-panel">
            <div class="panel-heading"><h2>Pass Rate Trend</h2><span>{len(pass_rates)} iteration(s)</span></div>
            <div class="line-chart">
                <svg viewBox="0 0 {width} {height}" role="img" aria-label="Pass rate by iteration">
                    <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" class="axis" />
                    <line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" class="axis" />
                    <line x1="{pad}" y1="{pad}" x2="{width - pad}" y2="{pad}" class="grid" />
                    <line x1="{pad}" y1="{height / 2:.1f}" x2="{width - pad}" y2="{height / 2:.1f}" class="grid" />
                    <polygon points="{area}" class="trend-area" />
                    <polyline points="{polyline}" class="trend-line" />
                    <g class="trend-points">{markers}</g>
                </svg>
                <div class="x-labels">{labels}</div>
            </div>
        </section>
    '''


def _safe_markdown(report: str) -> str:
    import markdown
    return markdown.markdown(escape(report), extensions=["tables", "fenced_code"])


def _build_html_report(report: str, state: QAState) -> str:
    metrics = _report_metrics(state)
    html_content = _safe_markdown(report)
    sut = state.get("sut_description", "Autonomous QA System")
    domain = state.get("domain", "AI quality assurance")
    iteration = state.get("current_iteration", 1)
    kpis = "".join([
        _kpi("Current Pass Rate", _pct(metrics["current_pass"]), f'{metrics["delta"] * 100:+.1f} pts from first iteration', "success"),
        _kpi("Coverage Score", _pct(metrics["coverage"]), "Estimated scenario coverage", "info"),
        _kpi("Tests Generated", str(metrics["tests"]), f'{metrics["verdicts"]} judged results', "neutral"),
        _kpi("Open Findings", str(metrics["failures"]), "Fail, error, and partial verdicts", "danger"),
        _kpi("Local Quality", _pct(metrics["local_quality"]), f'Risk avg: {_pct(metrics["local_risk"])}', "warning"),
        _kpi("Retrieval", f'{metrics["avg_contexts"]:.1f}', f'Avg context relevance: {_pct(metrics["context_relevance"])}', "info"),
        _kpi("Security Alerts", str(metrics["security_alerts"]), "Local injection/leakage flags", "danger"),
    ])
    palette = ["#3a86ff", "#21a67a", "#ff9f1c", "#ef476f", "#7c3aed"]
    charts = "\n".join([
        _line_chart(metrics["pass_rates"]),
        _donut_chart("Verdict Mix", metrics["statuses"]),
        _bar_chart("Severity Breakdown", metrics["severities"], ["#ef476f", "#ff9f1c", "#ffd166", "#21a67a", "#7c8a9a"], "No severity data available."),
        _bar_chart("Failure Categories", metrics["categories"], palette, "No failure categories found."),
        _bar_chart("Edge Case Coverage", metrics["edge_cases"], palette, "No test-case categories available."),
        _bar_chart("Local Evaluator Signals", metrics["local_recommendations"], palette, "No local evaluator data available."),
        _bar_chart("Grounding Verdicts", metrics["grounding"], palette, "No hallucination detector data available."),
    ])
    patterns = "".join(f"<li>{escape(str(pattern))}</li>" for pattern in metrics["patterns"][:8]) or "<li>No recurring failure patterns detected.</li>"
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Autonomous QA Report</title>
<style>
:root{{--text:#eef3ff;--muted:#9aa8bd;--line:#263244;--green:#21a67a;--blue:#3a86ff;--red:#ef476f;--amber:#ff9f1c}}*{{box-sizing:border-box}}body{{margin:0;color:var(--text);background:radial-gradient(circle at 12% 0%,rgba(58,134,255,.22),transparent 30%),radial-gradient(circle at 88% 8%,rgba(33,166,122,.16),transparent 28%),linear-gradient(135deg,#08111f 0%,#101827 44%,#161522 100%);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.6}}.report-shell{{width:min(1180px,calc(100% - 32px));margin:0 auto;padding:32px 0 56px}}.hero{{padding:32px;border:1px solid rgba(255,255,255,.12);border-radius:8px;background:linear-gradient(135deg,rgba(17,24,42,.92),rgba(21,31,52,.84));box-shadow:0 24px 70px rgba(0,0,0,.32)}}.eyebrow{{color:#8fd9ff;font-size:.78rem;font-weight:800;letter-spacing:0;text-transform:uppercase}}h1{{max-width:920px;margin:10px 0 12px;font-size:clamp(2rem,5vw,4.4rem);line-height:1.02;letter-spacing:0}}.hero p{{max-width:780px;margin:0;color:var(--muted);font-size:1.06rem}}.meta-row{{display:flex;flex-wrap:wrap;gap:10px;margin-top:22px}}.meta-pill{{padding:8px 12px;border:1px solid rgba(255,255,255,.13);border-radius:999px;background:rgba(255,255,255,.06);color:#d6deed;font-size:.88rem}}.kpi-grid{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:14px;margin:18px 0}}.kpi{{min-height:138px;padding:18px;border:1px solid rgba(255,255,255,.10);border-radius:8px;background:rgba(17,24,42,.82)}}.kpi span,.panel-heading span,.markdown-body th{{color:var(--muted)}}.kpi strong{{display:block;margin:8px 0 4px;font-size:2rem;line-height:1}}.kpi small{{color:#b8c2d6}}.kpi-success{{border-top:4px solid var(--green)}}.kpi-info{{border-top:4px solid var(--blue)}}.kpi-danger{{border-top:4px solid var(--red)}}.kpi-warning{{border-top:4px solid var(--amber)}}.charts-grid{{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:18px}}.panel{{border:1px solid rgba(255,255,255,.10);border-radius:8px;background:rgba(17,24,42,.86);box-shadow:0 18px 40px rgba(0,0,0,.2)}}.trend-panel{{grid-column:span 8}}.donut-panel{{grid-column:span 4}}.chart-panel{{grid-column:span 4}}.panel-heading{{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:18px 20px 0}}.panel h2{{margin:0;font-size:1.05rem}}.line-chart{{padding:10px 18px 20px}}.line-chart svg{{width:100%;height:auto;overflow:visible}}.axis{{stroke:#526074;stroke-width:1.5}}.grid{{stroke:#263244;stroke-width:1;stroke-dasharray:5 8}}.trend-area{{fill:rgba(58,134,255,.18)}}.trend-line{{fill:none;stroke:#5cb4ff;stroke-width:5;stroke-linecap:round;stroke-linejoin:round}}.trend-points circle{{fill:#fff;stroke:#3a86ff;stroke-width:4}}.x-labels{{position:relative;height:20px;margin:-8px 34px 0;color:var(--muted);font-size:.78rem}}.x-labels span{{position:absolute;transform:translateX(-50%)}}.donut-layout{{display:grid;grid-template-columns:180px 1fr;gap:12px;align-items:center;padding:14px 18px 20px}}.donut-layout svg{{width:180px;height:180px}}.donut-value{{fill:var(--text);font-size:34px;font-weight:800}}.donut-label{{fill:var(--muted);font-size:13px}}.legend{{list-style:none;padding:0;margin:0;color:#cbd5e1}}.legend li{{display:flex;align-items:center;gap:8px;margin:8px 0}}.legend span{{width:10px;height:10px;border-radius:999px;flex:0 0 auto}}.bar-chart{{padding:16px 20px 20px}}.bar-row{{margin-bottom:16px}}.bar-meta{{display:flex;justify-content:space-between;gap:12px;margin-bottom:7px;color:#dce5f4}}.bar-track{{height:10px;border-radius:999px;background:#263244;overflow:hidden}}.bar-fill{{height:100%;border-radius:999px}}.empty-chart{{padding:18px 20px 24px;color:var(--muted)}}.patterns{{margin:18px 0;padding:22px 24px}}.patterns h2{{margin:0 0 12px}}.patterns ul{{margin:0;padding-left:20px;color:#d7e0ef}}.markdown-body{{margin-top:18px;padding:28px;color:#dbe5f5}}.markdown-body h1,.markdown-body h2,.markdown-body h3{{color:#fff;line-height:1.2}}.markdown-body h1{{font-size:2rem;border-bottom:1px solid var(--line);padding-bottom:10px}}.markdown-body h2{{margin-top:30px;font-size:1.45rem}}.markdown-body p,.markdown-body li{{color:#d5dfef}}.markdown-body table{{display:block;width:100%;overflow-x:auto;border-collapse:collapse;margin:18px 0}}.markdown-body th,.markdown-body td{{padding:10px 12px;border:1px solid #2a3850;vertical-align:top}}.markdown-body th{{background:#172238;color:#f3f7ff}}.markdown-body tr:nth-child(even) td{{background:rgba(255,255,255,.025)}}.markdown-body code,.markdown-body pre{{background:#07111f;border:1px solid #25324a;border-radius:6px}}.markdown-body code{{padding:2px 5px}}.markdown-body pre{{padding:14px;overflow-x:auto}}@media(max-width:980px){{.kpi-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.trend-panel,.donut-panel,.chart-panel{{grid-column:1/-1}}}}@media(max-width:640px){{.report-shell{{width:min(100% - 20px,1180px);padding-top:10px}}.hero,.markdown-body{{padding:20px}}.kpi-grid{{grid-template-columns:1fr}}.donut-layout{{grid-template-columns:1fr;justify-items:start}}}}
</style>
</head>
<body><main class="report-shell"><section class="hero"><div class="eyebrow">Autonomous QA Intelligence Report</div><h1>Multi-Agent QA Results</h1><p>{escape(str(sut))}</p><div class="meta-row"><span class="meta-pill">Domain: {escape(str(domain))}</span><span class="meta-pill">Iteration: {iteration}</span><span class="meta-pill">Generated from LangGraph agent state</span></div></section><section class="kpi-grid">{kpis}</section><section class="charts-grid">{charts}</section><section class="panel patterns"><h2>Key Failure Patterns</h2><ul>{patterns}</ul></section><article class="panel markdown-body">{html_content}</article></main></body>
</html>'''


def reporter_node(state: QAState) -> dict:
    """LangGraph node: Reporter Agent. Generates final QA report."""
    iteration = state.get("current_iteration", 1)

    print(f"\n{'='*60}")
    print(f"  REPORTER AGENT - Final Report")
    print(f"{'='*60}")
    print(f"  Compiling report from {iteration} iteration(s)...")

    test_suite = state.get("test_suite", [])
    all_verdicts = state.get("all_verdicts", [])
    failure_patterns = state.get("failure_patterns", [])
    pass_rates = state.get("iteration_pass_rates", [])
    coverage_score = state.get("coverage_score", 0.0)

    report_prompt = REPORTER_GENERATION_PROMPT.format(
        sut_description=state.get("sut_description", "A RAG/LLM system"),
        domain=state.get("domain", "general"),
        total_iterations=iteration,
        pass_rates=", ".join(f"{r:.1%}" for r in pass_rates) if pass_rates else "N/A",
        all_test_cases=json.dumps(test_suite[:30], indent=2),  # Limit to avoid token overflow
        all_verdicts=json.dumps(all_verdicts[:30], indent=2),
        failure_patterns="\n".join(f"- {p}" for p in failure_patterns) if failure_patterns else "None",
        coverage_score=f"{coverage_score:.1%}",
    )

    llm = _get_llm()
    messages = [
        SystemMessage(content=REPORTER_SYSTEM_PROMPT),
        HumanMessage(content=report_prompt),
    ]
    response = llm.invoke(messages)

    report = response.content
    print(f"  Report generated ({len(report)} characters)")

    # Save report to file
    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/qa_report_iter{iteration}.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved Markdown to: {report_path}")

    # Generate HTML report and open in browser
    try:
        import webbrowser

        html_template = _build_html_report(report, state)
        
        html_path = f"reports/qa_report_iter{iteration}.html"
        with open(html_path, "w") as f:
            f.write(html_template)
        print(f"  Saved HTML to: {html_path}")
        
        # Automatically open in the user's default browser
        abs_path = os.path.abspath(html_path)
        webbrowser.open(f"file://{abs_path}")
        print(f"  🌐 Launched HTML report in your browser!")
    except ImportError:
        print("  ⚠️ Could not generate HTML report because the 'markdown' package is missing.")
    except Exception as e:
        print(f"  ⚠️ Could not open HTML report: {e}")

    return {"final_report": report}
