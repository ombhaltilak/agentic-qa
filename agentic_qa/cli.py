"""
Command-line interface.

    agentic-qa run --endpoint http://localhost:8000/chat --description "HR policy bot" \
        --iterations 3 --tests 8 --fail-under 0.8 --max-critical 0 --output qa_run.json

    agentic-qa run --target mypkg.rag:answer --replay qa_run.json --fail-under 0.9
    agentic-qa run --target mypkg.rag:answer --dataset golden.jsonl --max-hallucination 0.05
    agentic-qa bugs --runs 3            # planted-bug catch rate on the demo SUT
    agentic-qa label reports/qa_run_iter3.json --out labels.jsonl
    agentic-qa calibrate --dataset labels.jsonl
    agentic-qa diff baseline.json candidate.json --markdown diff.md --fail-on-new-failures

Exit codes: 0 = passed all gates, 1 = a quality gate failed, 2 = usage/runtime error.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys


def _load_callable(spec: str):
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError("--target must look like 'package.module:function'")
    sys.path.insert(0, os.getcwd())
    obj = importlib.import_module(module_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    if not (callable(obj) or hasattr(obj, "invoke")):
        raise ValueError(f"{spec} is neither callable nor a LangChain Runnable")
    return obj


def _parse_headers(items: list[str] | None) -> dict:
    headers = {}
    for item in items or []:
        name, sep, value = item.partition(":")
        if not sep or not name.strip():
            raise ValueError(f"--header must look like 'Name: value' (got {item!r})")
        headers[name.strip()] = value.strip()
    return headers


def _parse_body_template(raw: str | None):
    if not raw:
        return None
    from pathlib import Path

    text = Path(raw[1:]).read_text() if raw.startswith("@") else raw
    template = json.loads(text)
    if "{input}" not in json.dumps(template):
        raise ValueError("--body-template must contain the placeholder {input}")
    return template


DEFAULT_MAX_CRITICAL = 0  # with no gate flags, any critical/high failure fails the run


def _gate_args_given(args) -> bool:
    return any([args.fail_under is not None, args.max_critical is not None, args.fail_on_regression,
                args.max_policy_violations is not None, args.max_hallucination is not None])


def _evaluate_gates(summary: dict, args) -> tuple[list[str], list[str]]:
    """Returns (failures, gates applied). A run never "passes" without saying what was checked."""
    failures, applied = [], []
    max_critical = args.max_critical
    if not getattr(args, "no_gates", False) and not _gate_args_given(args):
        max_critical = DEFAULT_MAX_CRITICAL
        applied.append(f"critical/high failures <= {DEFAULT_MAX_CRITICAL} (default; set your own gates or --no-gates)")
    elif max_critical is not None:
        applied.append(f"critical/high failures <= {max_critical}")
    if args.fail_under is not None:
        applied.append(f"pass rate >= {args.fail_under:.0%}")
        if summary["pass_rate"] < args.fail_under:
            failures.append(f"pass rate {summary['pass_rate']:.1%} < required {args.fail_under:.1%}")
    if max_critical is not None and summary["critical_or_high_failures"] > max_critical:
        failures.append(f"{summary['critical_or_high_failures']} critical/high failures > allowed {max_critical}")
    if args.fail_on_regression:
        applied.append("no remembered failure still failing")
        if summary["regression_failures"] > 0:
            failures.append(f"{summary['regression_failures']} previously-seen failure(s) still failing")
    if args.max_policy_violations is not None:
        applied.append(f"policy-violating tests <= {args.max_policy_violations}")
        if summary["policy_violation_tests"] > args.max_policy_violations:
            failures.append(f"{summary['policy_violation_tests']} test(s) violated policy > allowed {args.max_policy_violations}")
    if args.max_hallucination is not None:
        applied.append(f"hallucination rate <= {args.max_hallucination:.0%}")
        gold = summary.get("golden")
        if not gold:
            failures.append("--max-hallucination needs a golden --dataset")
        elif gold["hallucination_rate"] > args.max_hallucination:
            failures.append(f"hallucination rate {gold['hallucination_rate']:.1%} > allowed {args.max_hallucination:.1%}")
    return failures, applied


def _print_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("  AGENTIC QA — RUN SUMMARY")
    print("=" * 60)
    print(f"  Tests: {summary['total_tests']}   Passed: {summary['passed']}   Failed: {summary['failed']}")
    print(f"  Pass rate: {summary['pass_rate']:.1%}   Coverage: {summary['coverage']:.1%}")
    print(f"  Critical/high failures: {summary['critical_or_high_failures']}   "
          f"Regression failures: {summary['regression_failures']}")
    decided = summary.get("decided_by", {})
    if decided.get("jev"):
        print(f"  Decided by Jev: {decided['jev']}   by LLM judge: {decided.get('llm', 0)}")
    if summary.get("invalid_tests"):
        print(f"  Invalid tests (wrong expectation; excluded from all metrics): {summary['invalid_tests']}")
    print(f"  Assertion overrides: {summary['assertion_overrides']}   "
          f"Needs human review: {summary['needs_human_review']}")
    lat, use = summary["sut_latency"], summary["usage"]
    if lat["samples"]:
        print(f"  SUT latency p50/p95/max: {lat['p50_s']:.3f}s / {lat['p95_s']:.3f}s / {lat['max_s']:.3f}s")
    print(f"  QA cost: ${use['estimated_cost_usd']:.4f} ({use['input_tokens']:,} in / "
          f"{use['output_tokens']:,} out tokens, {use['calls']} calls)")
    gold = summary.get("golden")
    if gold:
        print(f"  Golden: accuracy {gold['accuracy']:.1%} on {gold['items']} items   "
              f"HALLUCINATION RATE {gold['hallucination_rate']:.1%}")
        if gold.get("correct_but_policy_violation_ids"):
            print(f"          {len(gold['correct_but_policy_violation_ids'])} correct answer(s) still failed a policy rule "
                  f"({gold['passed_all_checks']}/{gold['items']} passed every check)")
    pol = summary.get("policy")
    if pol:
        print("\n  Policy rule            Targeted  Violations")
        for rid, s in pol.items():
            print(f"  {rid:22s} {s['targeted_by']:8d}  {s['violations']:10d}")
    cats = summary.get("categories", {})
    if cats:
        print("\n  Category             Tests  Pass rate")
        for cat, s in sorted(cats.items()):
            rate = "—" if s["pass_rate"] is None else f"{s['pass_rate']:.0%}"
            print(f"  {cat:20s} {s['tests']:5d}  {rate:>9s}")


def cmd_run(args) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    os.environ.setdefault("AUTO_OPEN_REPORT", "0")
    if args.reports_dir:
        os.environ["REPORTS_DIR"] = args.reports_dir
    if args.validity_model:
        os.environ["JUDGE_VALIDITY_MODEL"] = args.validity_model

    import agentic_qa
    from agentic_qa.sut.financial_rag import FinancialDocumentRAG

    kwargs = dict(
        system_name=args.name,
        system_description=args.description,
        domain=args.domain,
        max_iterations=args.iterations,
        tests_per_iteration=args.tests,
        model_name=args.model,
        judge_model=args.judge_model,
        judge_backend=args.judge,
        replay_file=args.replay or (args.dataset if args.no_generate else None),
        dataset_file=None if args.no_generate else args.dataset,
        policy_file=args.policy,
        enable_ragas=args.ragas,
        max_workers=args.workers,
        save_path=args.output,
    )
    if args.openai_chat:
        kwargs["adapter"] = agentic_qa.OpenAIChatAdapter(
            base_url=args.openai_chat, model=args.chat_model, description=args.description,
            system_name=args.name, domain=args.domain, api_key_env=args.api_key_env or None,
            system_prompt=args.system_prompt, timeout=args.timeout,
        )
    elif args.endpoint:
        kwargs["adapter"] = agentic_qa.APIAdapter(
            endpoint=args.endpoint, description=args.description, system_name=args.name,
            domain=args.domain, input_key=args.input_key, output_key=args.output_key,
            output_path=args.output_path, contexts_path=args.contexts_path,
            body_template=_parse_body_template(args.body_template),
            headers=_parse_headers(args.header), timeout=args.timeout,
        )
    elif args.target:
        obj = _load_callable(args.target)
        if agentic_qa._is_runnable(obj):
            kwargs["adapter"] = agentic_qa.LangChainAdapter(
                obj, description=args.description, system_name=args.name, domain=args.domain,
                input_key=args.chain_input_key or None)
        else:
            kwargs["target_function"] = obj
    else:
        demo = FinancialDocumentRAG()
        kwargs["adapter"] = demo
        kwargs["system_description"] = demo.describe()
        kwargs["domain"] = demo.domain

    final_state = agentic_qa.run_autonomous_test(**kwargs)
    summary = agentic_qa.run_summary(final_state)
    _print_summary(summary)

    if args.json_summary:
        print(json.dumps(summary, indent=2))

    failures, applied = _evaluate_gates(summary, args)
    if not applied:
        print("\n  ⚠️  No quality gates applied (--no-gates): exit code does not reflect quality")
        return 0
    print("\n  Gates: " + "; ".join(applied))
    if failures:
        print("  ❌ QUALITY GATE FAILED:")
        for f in failures:
            print(f"     - {f}")
        return 1
    print("  ✅ Quality gates passed")
    return 0


def cmd_bugs(args) -> int:
    """Run the demo SUT and report how many of its 6 planted bugs were caught."""
    from dotenv import load_dotenv

    load_dotenv()
    os.environ.setdefault("AUTO_OPEN_REPORT", "0")
    import agentic_qa
    from agentic_qa.benchmarks import evaluate_planted_bugs
    from agentic_qa.sut.financial_rag import FinancialDocumentRAG

    import tempfile

    runs = []
    for i in range(1, args.runs + 1):
        print(f"\n######## Planted-bug run {i}/{args.runs} ########")
        # Independent runs: fresh, isolated regression memory each time.
        os.environ["REGRESSION_MEMORY_PATH"] = os.path.join(tempfile.mkdtemp(prefix="aqa_bugs_"), "memory.json")
        demo = FinancialDocumentRAG()
        state = agentic_qa.run_autonomous_test(
            adapter=demo, system_description=demo.describe(), domain=demo.domain,
            max_iterations=args.iterations, tests_per_iteration=args.tests,
            model_name=args.model, judge_backend=args.judge, judge_model=args.judge_model,
        )
        result = evaluate_planted_bugs(state)
        result["cost_usd"] = agentic_qa.run_summary(state)["usage"]["estimated_cost_usd"]
        runs.append(result)

    print("\n" + "=" * 60)
    print("  PLANTED-BUG BENCHMARK (demo: Financial Document RAG)")
    print("=" * 60)
    print(f"  {'Bug':20s} " + " ".join(f"run{i + 1:<3d}" for i in range(len(runs))))
    for key, first in runs[0]["per_bug"].items():
        marks = ["caught" if r["per_bug"][key]["caught"] else ("exposed" if r["per_bug"][key]["exposed"] else "-")
                 for r in runs]
        print(f"  {key:20s} " + " ".join(f"{m:6s}" for m in marks))
    caught = [r["bugs_caught"] for r in runs]
    print(f"\n  Caught per run: {caught}  (mean {sum(caught) / len(caught):.1f}/6)")
    print("  exposed = a test triggered the bug but the judge passed it; - = never triggered")
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(json.dumps({"runs": runs}, indent=2))
        print(f"  Saved: {args.output}")
    return 0


def cmd_diff(args) -> int:
    from agentic_qa.diff import compare_runs, load_run, to_markdown

    diff = compare_runs(load_run(args.before), load_run(args.after))
    md = to_markdown(diff, os.path.basename(args.before), os.path.basename(args.after))
    print(md)
    if args.markdown:
        from pathlib import Path

        Path(args.markdown).write_text(md)
        print(f"\nSaved Markdown: {args.markdown}")
    if args.json:
        from pathlib import Path

        Path(args.json).write_text(json.dumps(diff, indent=2))
    if args.fail_on_new_failures and diff["newly_failing"]:
        print(f"\n  ❌ {len(diff['newly_failing'])} test(s) newly failing")
        return 1
    return 0


def cmd_label(args) -> int:
    from agentic_qa.calibration import label_run

    added = label_run(args.run, args.out)
    print(f"\nLabeled {added} new example(s) -> {args.out}")
    return 0


def cmd_rejudge(args) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    from agentic_qa.diff import load_run
    from agentic_qa.policy import load_policy
    from agentic_qa.rejudge import rejudge_run

    if args.validity_model:
        os.environ["JUDGE_VALIDITY_MODEL"] = args.validity_model
    run = load_run(args.run)
    policy = load_policy(args.policy) if args.policy else None
    new = rejudge_run(run, judge_model=args.judge_model, policy=policy)
    from pathlib import Path

    Path(args.output).write_text(json.dumps(new, indent=2, default=str))
    s = new["summary"]
    print(f"Re-judged {len(new['verdicts'])} test(s) with {new['judge_model']}: pass rate {s['pass_rate']:.1%}, "
          f"cost ${s['usage']['estimated_cost_usd']:.4f} -> {args.output}")
    return 0


def cmd_agreement(args) -> int:
    from agentic_qa.diff import load_run
    from agentic_qa.rejudge import agreement, load_labels

    labels = load_labels(args.labels)
    reports = {path: agreement(load_run(path), labels) for path in args.runs}
    print(json.dumps(reports, indent=2))

    def fmt(v, ci):
        return "—" if v is None else f"{v:.0%} [{ci[0]:.0%}-{ci[1]:.0%}]" if ci else f"{v:.0%}"

    print(f"\n{'run':40s} {'judge':14s} {'n':>3s}  {'accuracy':16s} {'fail precision':16s} {'fail recall':16s}")
    for path, r in reports.items():
        print(f"{path[-40:]:40s} {str(r['judge_model'])[:14]:14s} {r['labeled']:3d}  "
              f"{fmt(r['accuracy'], r['accuracy_95ci']):16s} {fmt(r['failure_precision'], r['failure_precision_95ci']):16s} "
              f"{fmt(r['failure_recall'], r['failure_recall_95ci']):16s}")
    return 0


def cmd_memory(args) -> int:
    if args.system:
        os.environ["REGRESSION_MEMORY_SCOPE"] = args.system
    from agentic_qa.agents.regression_memory import (
        _status, invalidate_cases, load_regression_cases, memory_path, memory_stats,
    )

    if args.action == "list":
        cases = load_regression_cases()
        print(f"{memory_path()}: {memory_stats()}")
        for i, c in enumerate(cases, 1):
            print(f"{i:3d}. [{_status(c):9s}] x{c.get('recurrence_count', 1)} {str(c.get('input_data', ''))[:90]!r}")
        return 0
    if not args.selector:
        print("memory invalidate needs a test number (from `memory list`) or text from its input", file=sys.stderr)
        return 2
    marked = invalidate_cases(args.selector, args.reason or "")
    for c in marked:
        print(f"marked invalid: {str(c.get('input_data', ''))[:100]!r}")
    if not marked:
        print("no matching remembered test")
    return 0 if marked else 1


def cmd_calibrate(args) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    if args.judge_model:
        os.environ["JUDGE_MODEL_NAME"] = args.judge_model
    from agentic_qa.calibration import calibrate

    backends = tuple(args.backends.split(","))
    report = calibrate(args.dataset, backends, test_fraction=args.test_fraction, seed=args.seed)
    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentic-qa", description="Autonomous multi-agent QA for RAG & LLM apps")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Run an autonomous QA session")
    target = r.add_mutually_exclusive_group()
    target.add_argument("--endpoint", help="HTTP endpoint of the system under test")
    target.add_argument("--target", help="Python 'module:attr' — a function (str -> str|dict) or a LangChain Runnable")
    target.add_argument("--openai-chat", metavar="BASE_URL",
                        help="OpenAI-compatible chat endpoint base, e.g. http://localhost:11434/v1")
    r.add_argument("--name", default="Target System")
    r.add_argument("--description", default="A RAG question-answering system")
    r.add_argument("--domain", default="general")
    r.add_argument("--input-key", default="query", help="JSON key for the question (API mode)")
    r.add_argument("--output-key", default="output", help="JSON key for the answer (API mode)")
    r.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds (API mode)")
    r.add_argument("--output-path", help="Dotted path to the answer in the JSON response, e.g. data.answer")
    r.add_argument("--contexts-path", help="Dotted path to retrieved texts, e.g. data.sources.*.text")
    r.add_argument("--body-template", help='JSON request body with {input}, or @file.json')
    r.add_argument("--header", action="append", help="'Name: value'; ${ENV_VAR} is expanded. Repeatable")
    r.add_argument("--chat-model", default="gpt-4o-mini", help="Model for --openai-chat")
    r.add_argument("--system-prompt", help="System prompt for --openai-chat")
    r.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Env var holding the key for --openai-chat")
    r.add_argument("--chain-input-key", dest="chain_input_key", default="input",
                   help="Input key when --target is a LangChain Runnable (default: input)")
    r.add_argument("--iterations", type=int, default=3)
    r.add_argument("--tests", type=int, default=5, help="Tests generated per iteration")
    r.add_argument("--model", default=os.getenv("MODEL_NAME", "gpt-4o-mini"))
    r.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL_NAME"))
    r.add_argument("--validity-model", help="Model that reviews generated tests for wrong expectations "
                   "(default gpt-4o; JUDGE_VALIDITY_CHECK=0 turns the review off)")
    r.add_argument("--judge", choices=["auto", "llm", "jev", "cascade", "ensemble"], default=os.getenv("JUDGE_BACKEND", "auto"))
    r.add_argument("--replay", help="Re-run a saved suite instead of generating tests")
    r.add_argument("--dataset", help="Golden dataset (JSONL: question + reference_answer from your docs)")
    r.add_argument("--no-generate", action="store_true", help="With --dataset: run only the golden items")
    r.add_argument("--policy", help="Policy file (YAML/JSON) of must / must_not rules")
    r.add_argument("--ragas", action="store_true", help="Also compute RAGAS metrics")
    r.add_argument("--workers", type=int, default=4, help="Concurrent SUT calls")
    r.add_argument("--output", help="Write run JSON here")
    r.add_argument("--reports-dir", help="Directory for Markdown/HTML/JSON reports")
    r.add_argument("--json-summary", action="store_true", help="Print the summary as JSON")
    r.add_argument("--fail-under", type=float, help="Fail if final pass rate is below this (0-1)")
    r.add_argument("--max-critical", type=int, help="Fail if critical/high failures exceed this")
    r.add_argument("--fail-on-regression", action="store_true", help="Fail if a remembered failure still fails")
    r.add_argument("--max-hallucination", type=float, help="Fail if golden hallucination rate exceeds this (0-1)")
    r.add_argument("--max-policy-violations", type=int, help="Fail if more tests than this violate the policy")
    r.add_argument("--no-gates", action="store_true",
                   help="Don't apply the default gate (any critical/high failure fails the run)")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("calibrate", help="Measure judge accuracy against labeled examples")
    c.add_argument("--dataset", help="JSON list of labeled examples (default: built-in set)")
    c.add_argument("--backends", default="assertions,llm,jev")
    c.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL_NAME"))
    c.add_argument("--test-fraction", type=float, default=0.5, help="Held-out share for reporting")
    c.add_argument("--seed", type=int, default=0, help="Split seed")
    c.set_defaults(func=cmd_calibrate)

    b = sub.add_parser("bugs", help="Measure how many of the demo's 6 planted bugs are caught")
    b.add_argument("--runs", type=int, default=1)
    b.add_argument("--iterations", type=int, default=3)
    b.add_argument("--tests", type=int, default=8)
    b.add_argument("--model", default=os.getenv("MODEL_NAME", "gpt-4o-mini"))
    b.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL_NAME"))
    b.add_argument("--judge", choices=["auto", "llm", "jev", "cascade", "ensemble"],
                   default=os.getenv("JUDGE_BACKEND", "auto"))
    b.add_argument("--output", help="Write benchmark JSON here")
    b.set_defaults(func=cmd_bugs)

    d = sub.add_parser("diff", help="Compare two run files: newly failing, fixed, metric deltas")
    d.add_argument("before", help="Baseline run JSON")
    d.add_argument("after", help="Candidate run JSON")
    d.add_argument("--markdown", help="Write the diff as Markdown (e.g. for a PR comment)")
    d.add_argument("--json", help="Write the full diff as JSON")
    d.add_argument("--fail-on-new-failures", action="store_true", help="Exit 1 if any shared test newly fails")
    d.set_defaults(func=cmd_diff)

    lb = sub.add_parser("label", help="Hand-label a saved run's outputs for calibration")
    lb.add_argument("run", help="Run JSON (reports/qa_run_iterN.json)")
    lb.add_argument("--out", default="labels.jsonl")
    lb.set_defaults(func=cmd_label)

    rj = sub.add_parser("rejudge", help="Re-judge a saved run's outputs (e.g. with a stronger judge model)")
    rj.add_argument("run", help="Run JSON")
    rj.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL_NAME"))
    rj.add_argument("--validity-model", help="Model that reviews tests for wrong expectations (default gpt-4o)")
    rj.add_argument("--policy", help="Policy file (default: the one stored in the run)")
    rj.add_argument("--output", required=True, help="Write the re-judged run JSON here")
    rj.set_defaults(func=cmd_rejudge)

    ag = sub.add_parser("agreement", help="Score runs' verdicts against human labels (precision / recall)")
    ag.add_argument("runs", nargs="+", help="Run JSON file(s) to score")
    ag.add_argument("--labels", required=True, help="labels.jsonl from `agentic-qa label`")
    ag.set_defaults(func=cmd_agreement)

    m = sub.add_parser("memory", help="List remembered failures or mark a bad test as invalid")
    m.add_argument("action", choices=["list", "invalidate"])
    m.add_argument("selector", nargs="?", help="invalidate: number from `memory list`, or text from the test input")
    m.add_argument("--system", help="System name used with --name in `run` (memory is kept per system)")
    m.add_argument("--reason", help="Why the test is invalid (shown to the Red-Team)")
    m.set_defaults(func=cmd_memory)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        print(f"agentic-qa: error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
