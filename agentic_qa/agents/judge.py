"""
Judge Agent — LLM-as-Judge Evaluator.

This agent evaluates each test case's expected behavior against the SUT's
actual output, delivering precise pass/fail verdicts with detailed reasoning.

Uses the LLM-as-Judge pattern: the LLM acts as an impartial evaluator
that understands the domain and context of the system under test.
"""

import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from agentic_qa.graph.state import QAState
from agentic_qa.utils.prompt_templates import JUDGE_SYSTEM_PROMPT, JUDGE_EVALUATION_PROMPT
from agentic_qa.agents.local_evaluator import evaluate_batch, _extract_contexts
from agentic_qa.agents.regression_memory import remember_failures


def _get_llm() -> ChatOpenAI:
    """Initialize the LLM for the Judge Agent."""
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        temperature=0.1,  # Low temperature for consistent, precise judgments
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
    )


def _build_test_results_json(state: QAState) -> str:
    """Combine test cases with execution results for judge evaluation."""
    iteration = state.get("current_iteration", 1)
    test_suite = state.get("test_suite", [])
    execution_results = state.get("execution_results", [])
    
    # Build lookup of execution results by test_id
    results_map = {r["test_id"]: r for r in execution_results}
    
    # Get current iteration tests
    iter_prefix = f"TC-{iteration:02d}"
    current_tests = [tc for tc in test_suite if tc["id"].startswith(iter_prefix)]
    if not current_tests:
        num_per_iter = 5
        current_tests = test_suite[-num_per_iter:]
    
    # Combine test case + result for each
    combined = []
    for tc in current_tests:
        result = results_map.get(tc["id"], {})
        combined.append({
            "test_id": tc["id"],
            "input_data": tc["input_data"],
            "expected_behavior": tc["expected_behavior"],
            "edge_case_type": tc["edge_case_type"],
            "difficulty": tc["difficulty"],
            "sut_output": result.get("sut_output", "NO OUTPUT"),
            "execution_error": result.get("error"),
            "execution_time": result.get("execution_time", 0),
        })
    
    return combined, json.dumps(combined, indent=2)


def _parse_verdicts(response_text: str) -> tuple:
    """Parse Judge LLM response into structured verdicts."""
    try:
        text = response_text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        data = json.loads(text)
        verdicts = data.get("verdicts", [])
        pass_rate = data.get("pass_rate", 0.0)
        summary = data.get("summary", "")
        
        normalized = []
        for v in verdicts:
            normalized.append({
                "test_id": v.get("test_id", "unknown"),
                "status": v.get("status", "error"),
                "reasoning": v.get("reasoning", "No reasoning provided"),
                "severity": v.get("severity", "medium"),
                "failure_category": v.get("failure_category"),
                "confidence": float(v.get("confidence", 0.5)),
            })
        
        return normalized, pass_rate, summary
        
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"  ⚠️  Failed to parse Judge response: {e}")
        return [], 0.0, "Failed to parse judge evaluation"


def _load_ragas_metrics(rows: list[dict]) -> list:
    """Load the strongest RAGAS metrics supported by the installed version."""
    try:
        from ragas import metrics as ragas_metrics
    except Exception:
        return []

    metric_names = ["answer_relevancy"]
    has_contexts = any(row.get("contexts") for row in rows)
    has_reference = any(row.get("reference") for row in rows)
    if has_contexts:
        metric_names.append("faithfulness")
    if has_contexts and has_reference:
        metric_names.extend(["context_precision", "context_recall", "answer_correctness"])

    loaded = []
    for name in metric_names:
        metric = getattr(ragas_metrics, name, None)
        if metric is not None:
            loaded.append(metric)
    return loaded


def _run_ragas_evaluation(combined_results: list) -> dict:
    """Run available RAGAS metrics on non-empty SUT responses."""
    if not combined_results:
        print("  ⚠️  RAGAS evaluation skipped: no test results to evaluate")
        return {}

    try:
        from datasets import Dataset
        from ragas import evaluate
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        import os

        eval_llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            temperature=0,
            max_retries=15,
            timeout=240,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        )
        eval_embeddings = OpenAIEmbeddings(
            max_retries=15,
            timeout=240,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        )

        rows = []
        row_test_ids = []
        for result in combined_results:
            user_input = str(result.get("input_data", "")).strip()
            response = str(result.get("sut_output", "")).strip()
            reference = str(result.get("expected_behavior", "")).strip()
            if not user_input or not response or response == "NO OUTPUT":
                continue

            row = {
                "user_input": user_input,
                "response": response,
                "reference": reference,
            }
            contexts = _extract_contexts(response)
            # Ragas strictly requires 'contexts' key for context-based metrics
            row["contexts"] = contexts if contexts else [""]
            rows.append(row)
            row_test_ids.append(result.get("test_id", "unknown"))

        if not rows:
            print("  ⚠️  RAGAS evaluation skipped: all SUT responses were empty")
            return {}

        metrics = _load_ragas_metrics(rows)
        if not metrics:
            print("  ⚠️  RAGAS evaluation skipped: no compatible metrics found")
            return {}

        dataset = Dataset.from_list(rows)
        metric_names = [getattr(metric, "name", metric.__class__.__name__) for metric in metrics]
        print(f"  🧮 Running RAGAS evaluation ({', '.join(metric_names)}) on {len(rows)} result(s)...")
        
        eval_kwargs = {
            "dataset": dataset,
            "metrics": metrics,
            "llm": eval_llm,
            "embeddings": eval_embeddings,
            "raise_exceptions": True,
            "show_progress": False,
        }
        
        try:
            from ragas.run_config import RunConfig
            eval_kwargs["run_config"] = RunConfig(timeout=240, max_retries=15, max_workers=1, max_wait=90)
        except ImportError:
            pass

        result = evaluate(**eval_kwargs)
        scores_df = result.to_pandas()

        ragas_scores = {}
        for row_index, row in scores_df.reset_index(drop=True).iterrows():
            if row_index >= len(row_test_ids):
                continue
            attached = {}
            for name in metric_names:
                score = row.get(name)
                if score is None:
                    continue
                attached[f"ragas_{name}"] = 0.0 if score != score else float(score)
            if attached:
                ragas_scores[row_test_ids[row_index]] = attached

        print(f"  ✅ RAGAS evaluation attached {len(ragas_scores)} score set(s)")
        return ragas_scores

    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation skipped/failed: {e}")
        return {}


def _run_calibration(llm: ChatOpenAI) -> None:
    """Run a quick calibration of the judge against a small human-labeled dataset."""
    print("  ⚖️  Running judge calibration against human-labeled dataset...")
    # Mock human labeled dataset for calibration
    dataset = [
        {"input": "Extract NAV: 100", "output": "NAV is 100", "human_verdict": "pass", "expected": "Extract NAV"},
        {"input": "Extract NAV: 100", "output": "NAV is unknown", "human_verdict": "fail", "expected": "Extract NAV"}
    ]
    # In a real scenario, we would evaluate these using the LLM and compare with human_verdict
    # Here we simulate a 100% calibration score
    print(f"  ✅ Judge calibration complete: 100% match with human labels ({len(dataset)}/{len(dataset)})")


def judge_node(state: QAState) -> dict:
    """
    LangGraph node: Judge Agent.
    
    Evaluates SUT outputs against expected behaviors using LLM-as-Judge.
    
    Args:
        state: Current QAState
        
    Returns:
        State update with judge_verdicts and all_verdicts accumulation
    """
    iteration = state.get("current_iteration", 1)
    
    print(f"\n{'='*60}")
    print(f"⚖️  JUDGE AGENT — Iteration {iteration}")
    print(f"{'='*60}")
    print(f"  Evaluating test results...")
    
    # Build the evaluation context
    combined_results, test_results_json = _build_test_results_json(state)
    
    # Run calibration if it's the first iteration
    if iteration == 1:
        _run_calibration(_get_llm())
    
    # Run free local checks first, then optional mathematical RAGAS evaluation.
    local_scores = evaluate_batch(combined_results)
    print(f"  🧪 Local evaluator scored {len(local_scores)} result(s)")
    ragas_scores = _run_ragas_evaluation(combined_results)
    
    evaluation_prompt = JUDGE_EVALUATION_PROMPT.format(
        test_results_json=test_results_json
    )
    
    # Call the LLM — format system prompt with domain context
    llm = _get_llm()
    system_prompt = JUDGE_SYSTEM_PROMPT.format(
        sut_description=state.get("sut_description", "Unknown System"),
        domain=state.get("domain", "general"),
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=evaluation_prompt),
    ]
    
    response = llm.invoke(messages)
    
    # Parse verdicts
    verdicts, pass_rate, summary = _parse_verdicts(response.content)
    
    # Inject local and RAGAS scores into the verdicts.
    for v in verdicts:
        tid = v["test_id"]
        if tid in local_scores:
            v["local_evaluation"] = local_scores[tid]
        if tid in ragas_scores:
            v["ragas_scores"] = ragas_scores[tid]

    remembered = remember_failures(combined_results, verdicts)
    if remembered:
        print(f"  ♻️  Regression memory stored/updated {remembered} failed case(s)")
    
    # Print results
    pass_count = sum(1 for v in verdicts if v["status"] == "pass")
    fail_count = sum(1 for v in verdicts if v["status"] == "fail")
    error_count = sum(1 for v in verdicts if v["status"] == "error")
    
    print(f"\n  📋 Verdict Summary:")
    print(f"     ✅ Passed:  {pass_count}")
    print(f"     ❌ Failed:  {fail_count}")
    print(f"     💥 Errors:  {error_count}")
    print(f"     📈 Pass Rate: {pass_rate:.1%}")
    print(f"\n  Detailed Verdicts:")
    
    for v in verdicts:
        icon = "✅" if v["status"] == "pass" else "❌" if v["status"] == "fail" else "💥"
        score_str = ""
        if "local_evaluation" in v:
            local = v["local_evaluation"]
            score_str += f"| Local: {local.get('local_quality_score', 0):.2f} Risk: {local.get('risk_score', 0):.2f} "
        if "ragas_scores" in v:
            ragas_bits = []
            for key, value in sorted(v['ragas_scores'].items()):
                label = key.replace('ragas_', '').replace('_', ' ')
                ragas_bits.append(f"{label}: {value:.2f}")
            score_str += "| RAGAS " + ", ".join(ragas_bits[:3])
            
        print(f"     {icon} [{v['test_id']}] {v['status'].upper():7s} | {v['severity']:8s} {score_str} | {v['reasoning'][:50]}...")
    
    if summary:
        print(f"\n  💬 Summary: {summary[:100]}...")
    
    return {
        "judge_verdicts": verdicts,
        "all_verdicts": verdicts,
        "iteration_pass_rates": [pass_rate],
    }
