import pandas as pd
from collections import Counter
from statistics import mean
from agentic_qa.agents.regression_memory import load_regression_cases

def display_qa_results(state: dict, num_regression_cases: int = 10) -> tuple:
    """
    Summarizes and displays the final QA metrics, DataFrames, and reports from the pipeline state.
    Designed for use in interactive environments like Jupyter Notebooks.
    
    Args:
        state: The final state dictionary returned by `run_qa_pipeline()`.
        num_regression_cases: Number of recent regression cases to load for the display.
        
    Returns:
        A tuple of (verdicts_df, regression_df) pandas DataFrames for further inspection.
    """
    try:
        from IPython.display import display, Markdown, HTML
        in_jupyter = True
    except ImportError:
        in_jupyter = False

    def _display_header(text: str):
        if in_jupyter:
            display(Markdown(f"### {text}"))
        else:
            print(f"\n{'='*40}\n{text}\n{'='*40}")

    # 1. Deduced Architecture
    _display_header("Deduced Architecture (Graphify)")
    if in_jupyter:
        display(Markdown(state.get("sut_architecture", "No architecture deduced.")))
    else:
        print(state.get("sut_architecture", "No architecture deduced."))

    # 2. Final Report summary
    _display_header("Final Autonomous Report")
    if in_jupyter:
        display(Markdown(state.get("final_report", "No report generated.")))
    else:
        print(state.get("final_report", "No report generated."))

    # 3. Calculate high-level metrics
    verdicts = state.get("all_verdicts", []) or state.get("judge_verdicts", [])
    local_evals = [v.get("local_evaluation", {}) for v in verdicts if v.get("local_evaluation")]
    ragas_scores = [v.get("ragas_scores", {}) for v in verdicts if v.get("ragas_scores")]
    
    grounding = Counter(
        e.get("hallucination_detection", {}).get("verdict", "unknown")
        for e in local_evals
    )
    
    avg_quality = mean([float(e.get("local_quality_score", 0.0)) for e in local_evals]) if local_evals else 0.0
    avg_risk = mean([float(e.get("risk_score", 0.0)) for e in local_evals]) if local_evals else 0.0
    avg_contexts = mean([
        e.get("retrieval_inspection", {}).get("contexts_found", 0)
        for e in local_evals
    ]) if local_evals else 0.0
    
    security_alerts = sum(1 for e in local_evals if e.get("security_flags"))
    ragas_metric_names = sorted({name for scores in ragas_scores for name in scores})
    
    # 4. Print High-level signals
    _display_header("Final QA Metrics & Signals")
    metrics_text = (
        f"**Coverage Score:** {state.get('coverage_score', 0):.0%}\n\n"
        f"**Local Quality Avg:** {avg_quality:.0%}\n\n"
        f"**Local Risk Avg:** {avg_risk:.0%}\n\n"
        f"**Avg Retrieved Contexts:** {avg_contexts:.1f}\n\n"
        f"**Security Alerts:** {security_alerts}\n\n"
        f"**Grounding Verdicts:** {dict(grounding)}\n\n"
        f"**RAGAS Metrics Attached:** {ragas_metric_names or 'none'}\n\n"
        f"**Total Regression Cases in Memory:** {len(load_regression_cases())}"
    )
    
    if in_jupyter:
        display(Markdown(metrics_text))
    else:
        print(metrics_text.replace("**", ""))

    # 5. Build Verdicts DataFrame
    rows = []
    for verdict in verdicts:
        local = verdict.get("local_evaluation", {}) or {}
        retrieval = local.get("retrieval_inspection", {}) or {}
        hallucination = local.get("hallucination_detection", {}) or {}
        rows.append({
            "test_id": verdict.get("test_id"),
            "status": verdict.get("status"),
            "severity": verdict.get("severity"),
            "confidence": verdict.get("confidence"),
            "local_quality": local.get("local_quality_score"),
            "risk": local.get("risk_score"),
            "contexts_found": retrieval.get("contexts_found"),
            "context_relevance": retrieval.get("context_relevance"),
            "grounding": hallucination.get("verdict"),
            "security_flags": len(local.get("security_flags", [])),
            "ragas_metrics": sorted((verdict.get("ragas_scores") or {}).keys()),
        })
    verdicts_df = pd.DataFrame(rows)

    # 6. Build Regression Memory DataFrame
    regression_cases = load_regression_cases(limit=num_regression_cases)
    regression_df = pd.DataFrame(regression_cases)

    # Display DataFrames
    _display_header("Detailed Test Verdicts")
    if in_jupyter:
        display(verdicts_df)
    else:
        print(verdicts_df.to_string())

    _display_header(f"Regression Memory (Last {num_regression_cases} cases)")
    if in_jupyter:
        display(regression_df)
    else:
        print(regression_df.to_string())

    return verdicts_df, regression_df
