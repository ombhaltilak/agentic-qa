# 🛡️ Agentic QA — Autonomous Multi-Agent Testing for RAG & LLMs

[![PyPI](https://img.shields.io/pypi/v/agentic-qa?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/agentic-qa/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-1C3C3C?logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/ombhaltilak/agentic-qa/blob/main/LICENSE)
[![CI](https://github.com/ombhaltilak/agentic-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/ombhaltilak/agentic-qa/actions/workflows/ci.yml)

**Agentic QA** is an open-source Python library that autonomously tests RAG pipelines and LLM applications — no manual test cases required.

It operates as a **self-improving red-team loop**: a team of LangGraph agents generates adversarial inputs, executes them against your system, evaluates the responses, and refines its attack strategy based on failures — all without human intervention.

> Built to solve the problem of manual QA in production RAG systems, where writing edge-case test suites is time-consuming and often incomplete.

---

## 🔍 What makes it different?

Metric libraries such as RAGAS score answers to test cases **you** provide. Agentic QA is the loop *around* that:

| Capability | Agentic QA |
|---|---|
| Test generation | Autonomous red-team guided by an 11-category attack taxonomy |
| Self-improvement | Refiner feeds failure patterns + weakest/untested categories back to the red-team |
| Trustworthy, cheap verdicts | Layered judge: deterministic assertions **override** model verdicts; **Jev** (free tier supported) judges first and only uncertain cases reach the LLM |
| Your rules, enforced | `policy.yaml` rules become regex checks, citation checks against retrieved chunks, judge criteria **and** red-team targets (every rule gets attacked) |
| Bad-test detection | A reviewer checks generated tests' expectations *without seeing the answer*; wrong ones become `invalid` and are excluded instead of counted as failures |
| Measured coverage | Coverage = taxonomy categories actually exercised, with per-category pass rates (not an LLM guess) |
| Regression memory | Failures persist across runs, are replayed, and are marked fixed when they pass |
| Reproducibility | Every run exports JSON; `--replay` re-executes the exact suite without generation |
| CI-ready | Exits non-zero on any critical/high failure by default; custom gates (`--fail-under`, `--max-hallucination`, …) and `agentic-qa diff` for before/after |
| Ground truth | Golden dataset of questions + reference answers from your documents → **hallucination rate** |
| Judge calibration | `agentic-qa label` + `calibrate` / `rejudge` / `agreement`: judge accuracy, failure precision and recall with 95% CIs |
| Measured detection | `agentic-qa bugs`: how many of the demo's 6 planted bugs the full loop catches |
| Latency & cost | SUT p50/p95 latency and the QA run's own token cost in every report |

---

## 🚀 Quick Start

### Installation

```bash
pip install agentic-qa
# or, to develop: git clone https://github.com/ombhaltilak/agentic-qa && cd agentic-qa && pip install -e ".[dev]"
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

### Test any RAG function in 5 lines

```python
import agentic_qa

def my_rag(query: str) -> str:
    return my_langchain_pipeline.invoke(query)

final_state = agentic_qa.run_autonomous_test(
    target_function=my_rag,
    system_name="My RAG App",
    system_description="A chatbot that answers questions using company documents.",
    domain="customer support",
    max_iterations=3,
    tests_per_iteration=5,
)

print(final_state["final_report"])
```

### Test a deployed REST API

```python
import agentic_qa

final_state = agentic_qa.run_autonomous_test(
    api_endpoint="http://localhost:8000/api/chat",
    system_name="Support Bot",
    system_description="An AI agent that resolves customer support tickets.",
    domain="customer support",
)
```

---

## 🧰 Command line & CI

```bash
# Test a running API, fail the build if quality regresses
agentic-qa run --endpoint http://localhost:8000/chat --input-key question --output-key answer \
  --description "HR policy assistant" --domain hr \
  --iterations 3 --tests 8 --output qa_run.json \
  --fail-under 0.8 --max-critical 0 --fail-on-regression

# Test a Python function (str -> str | {"answer": ..., "contexts": [...]})
agentic-qa run --target my_app.rag:answer --description "Docs chatbot"

# Re-run the exact saved suite (no generation; deterministic inputs) — e.g. after a fix
agentic-qa run --target my_app.rag:answer --replay qa_run.json --fail-under 0.9

# Compare two runs (newly failing / fixed / metric deltas)
agentic-qa diff baseline.json candidate.json --markdown diff.md

# Measure how accurate your judges are
agentic-qa label qa_run.json --out labels.jsonl          # hand-label outputs (judge verdict hidden)
agentic-qa calibrate --dataset labels.jsonl
agentic-qa rejudge qa_run.json --judge-model gpt-4o --output qa_run_gpt4o.json
agentic-qa agreement qa_run.json qa_run_gpt4o.json --labels labels.jsonl

# Inspect / clean regression memory
agentic-qa memory list --system "HR policy assistant"
```

Exit codes: `0` gates passed · `1` a quality gate failed · `2` usage/runtime error.
With no gate flags, the default gate fails the run on any critical/high failure, and the CLI
always prints which gates it checked. `--no-gates` turns this off (the exit code then says nothing about quality).
The CLI prints a per-category pass-rate table and writes Markdown, HTML and JSON reports to `reports/`.

---

## 🌍 Testing a real system

Two worked examples:

- [`examples/research_paper_rag/`](https://github.com/ombhaltilak/agentic-qa/blob/main/examples/research_paper_rag/): a FAISS + GPT-4o-mini RAG over *Attention Is All You Need*, with a golden set checked against the PDF and a policy file.
- [`examples/youtube_rag/`](https://github.com/ombhaltilak/agentic-qa/blob/main/examples/youtube_rag/): a Flask YouTube-video assistant tested over HTTP. Golden set checked against the transcript; the policy checks timestamp citations against the retrieved chunks. The run found that the assistant **invents timestamp ranges** in about a quarter of its answers, while answering 85–90% of golden questions correctly with no invented facts on unanswerable ones.

### Connect anything

| Your system | How |
|---|---|
| Python function `str -> str \| {"answer", "contexts"}` | `--target my_app.rag:answer` |
| LangChain Runnable (e.g. `create_retrieval_chain`) | `--target my_app.rag:rag_chain` (contexts read from the chain's documents; `--chain-input-key` if not `input`) |
| HTTP API with nested JSON | `--endpoint URL --output-path data.answer --contexts-path data.sources.*.text --body-template '{"query": {"text": "{input}"}}' --header 'Authorization: Bearer ${BOT_KEY}'` |
| OpenAI-compatible chat (OpenAI, vLLM, Ollama, LM Studio) | `--openai-chat http://localhost:11434/v1 --chat-model llama3.1 --system-prompt "..." --api-key-env OPENAI_API_KEY` |

`${VAR}` in headers is read from the environment, so secrets never appear on the command line or in shell history.

### Your rules: `policy.yaml`

```yaml
rules:
  - id: no_investment_advice
    type: must_not
    rule: recommend buying or selling a specific security
    severity: high
    patterns: ["you should (buy|sell)"]      # optional: deterministic, hard check
  - id: admits_unknown
    type: must
    rule: say it doesn't know when the documents don't contain the answer
```

Rules can also check citations against what the retriever returned:

```yaml
  - id: citations_exist
    type: must
    rule: only cite timestamp ranges that exist in the retrieved chunks
    citations: "\\[(\\d{1,2}:\\d{2}\\s*-\\s*\\d{1,2}:\\d{2})\\]"   # every match must appear in the contexts
```

Each rule is used three ways:
- its `patterns` / `citations` become a deterministic assertion (`hard: false` reports the violation without
  failing the test). For these rules the code decides; the LLM judge is told not to judge them, and its
  opinion on them is ignored
- its text goes to the LLM judge, which reports `policy_violations`, and to Jev as a P(violation) question
- the red team must break it: each iteration the least-tested rules are assigned, and a follow-up call
  writes a test for any rule the generator skipped (`POLICY_TESTS_PER_ITERATION`, default 2)

The report shows tests targeting each rule and violations found. Gate with `--max-policy-violations 0`.

### Did my change help? `agentic-qa diff`

```bash
agentic-qa run ... --output baseline.json
# change the prompt / model / chunking / retriever
agentic-qa run ... --replay baseline.json --output candidate.json
agentic-qa diff baseline.json candidate.json --markdown diff.md --fail-on-new-failures
```

The diff matches tests by question text and reports:
- tests that **newly fail** and tests that got **fixed**
- the pass rate on shared tests
- changes in hallucination rate, critical failures, policy violations, p50/p95 latency, cost and coverage
- categories whose pass rate moved

The Markdown output is ready to post as a PR comment.

---

## 🥇 Golden dataset & hallucination rate

Generated tests have LLM-written expectations. A golden set ties the run to ground truth:

```jsonl
{"question": "What was the NAV on 31 Mar 2024?", "reference_answer": "102.45", "source": "nav_report_2024.pdf p.2"}
{"question": "Who is the fund's custodian?", "answerable": false}
```

```bash
agentic-qa run --target my_app.rag:answer --dataset golden.jsonl            # golden + adversarial
agentic-qa run --target my_app.rag:answer --dataset golden.jsonl --no-generate \
  --max-hallucination 0.05                                                    # golden only, CI gate
```

- **Hallucination rate:** a wrong answer that is *not* an "I don't know", or any answer to an unanswerable question.
- Also reported: golden accuracy, abstention rate on answerable items (missed answers), and correct-abstention rate.
- RAGAS context recall and answer correctness use **only** golden reference answers, never LLM-written expectations.

Write reference answers from the source documents yourself. If an LLM drafts them, the metric inherits that LLM's errors.

---

## 🎯 Measuring the framework itself

**Planted bugs.** The demo system has 6 planted bugs: negative NAV, impossible dates, prompt injection, USD-only amounts, Unicode stripping, and allocations that don't sum to 100%. A deterministic detector checks each one. A bug is *caught* when a test triggered it and the judge failed that test:

```bash
agentic-qa bugs --runs 3 --iterations 3 --tests 8 --output bugs.json
```

**Judge calibration.** The built-in 12-example set is only a smoke test. One example moves accuracy by about 8 points. Label 50 or more real outputs yourself. The judge's verdict is hidden while you label, and you can stop and resume:

```bash
agentic-qa label reports/qa_run_iter3.json --out labels.jsonl
agentic-qa calibrate --dataset labels.jsonl            # tunes Jev's threshold on half, reports on the other half
```

Results are reported on the held-out half with 95% Wilson intervals and false-pass rate.

**Judge precision on a real run.** Re-judge the same saved outputs with another model (the SUT is not
called again) and score every run against your labels:

```bash
agentic-qa rejudge qa_run.json --judge-model gpt-4o --output qa_run_gpt4o.json
agentic-qa agreement qa_run.json qa_run_gpt4o.json --labels labels.jsonl
```

`agreement` reports accuracy, *failure precision* (of the failures reported, how many were real) and
*failure recall* (of the real failures, how many were caught), each with a 95% interval, plus the ids of
false alarms and missed failures. On the YouTube RAG example (38 outputs; labels in
`examples/youtube_rag/labels_claude.jsonl` were written by an AI assistant, not a human, so treat them as
a starting point) recall was 100% and failure precision 58% with a gpt-4o-mini judge and 73% with gpt-4o.
Most remaining false alarms come from generated tests with debatable expectations.

**Invalid tests.** Generated tests sometimes have a wrong expectation (e.g. "should say it's not in the
source" for a question the source answers). Before judging, a reviewer model checks each generated
test's expectation against the system description, policy and retrieved contexts. It does not see the
output, so the answer can't bias it. Tests with a wrong expectation are judged on their merits. If the output
is fine, the verdict is `invalid`: excluded from pass rate, failure counts, gates and diffs, retired
from regression memory, and shown to the red team as what not to write. A bad output still fails,
as does any deterministic or policy failure. Golden tests are never invalid.

The reviewer defaults to `gpt-4o` (`--validity-model`, `JUDGE_VALIDITY_MODEL`; about $0.10 per run of 40
tests). In our tests gpt-4o-mini accepted every expectation, including clearly wrong ones, so it is not
used for this. `JUDGE_VALIDITY_CHECK=0` turns the review off.

**Bad tests by hand.** When you spot a wrong expectation yourself, label it `[b]ad test` and remove it from memory:

```bash
agentic-qa memory list --system "My RAG"
agentic-qa memory invalidate 7 --system "My RAG" --reason "harmless question tagged as harmful"
```

Invalid tests are never replayed or re-remembered, and the red team is shown them as examples not to repeat.
Memory is kept per system (`--name`), in `reports/regression_memory/<name>.json`.

**Regression memory doesn't trust one verdict.** A first failure is stored as *suspected* and replayed in the next run. It becomes *confirmed* only on a second failure. A suspect that then passes is marked *flaky* and dropped. `--fail-on-regression` only counts cases that failed again.

**Latency & cost.** Each report includes SUT latency (p50/p95/max) and the QA run's own token usage and estimated cost per agent. Prices are estimates; override them with `LLM_PRICES_JSON='{"model": [in_per_1M, out_per_1M]}'`.

---

## ⚖️ Layered judging

Each result goes through independent layers. **A hard assertion failure overrides a passing model verdict.**

| Layer | Cost | What it catches |
|---|---|---|
| Deterministic assertions | free | crashes, empty output, PII leaks (email/SSN/Luhn-valid cards), injection compliance (echoed payloads, canaries), system-prompt disclosure, script markup |
| Local evaluator | free | keyword coverage, grounding vs. retrieved contexts, risk score |
| LLM judge | LLM | semantic correctness; use a stronger `--judge-model` than the generator |
| **Jev** decision model (optional) | **free** (`jev-1.13-free`) · paid $0.042 / 1M input | P(meets expected), P(grounded), P(complied with attack) — typed probabilities, not text |
| RAGAS (optional, `--ragas`) | LLM | faithfulness, context precision/recall, answer correctness/relevancy |

`--judge` / `JUDGE_BACKEND` selects the model judge:

- `cascade` — **Jev judges every test; only uncertain ones (P(pass) in 0.3–0.7) go to the LLM judge.** Tests already failed by a hard assertion skip the LLM entirely.
- `jev` — Jev only (no LLM judge calls; free on `jev-1.13-free`)
- `ensemble` — LLM judges everything, Jev adds probabilities; strong disagreements are marked `needs_human_review`
- `llm` — LLM-as-judge only
- `auto` (default) — `cascade` when a Jev key is configured, else `llm`

Why cascade rather than a vote: a 2026 study of decision models vs. LLM judges ([Rao & Callison-Burch, arXiv 2609.29769](https://arxiv.org/abs/2609.29769)) found Jev roughly matches flash-tier LLM judges on binary criteria at 16–325× lower cost, but the two make *correlated* errors — so a second model mostly saves money rather than adding accuracy. The deterministic assertions are the independent safety net.

### Using Jev for free

```bash
pip install "agentic-qa[jev]"
export OPENCODE_API_KEY=...      # OpenCode Zen key → uses jev-1.13-free
agentic-qa calibrate             # measure Jev's accuracy / Brier score on labeled data
agentic-qa run --target my_app.rag:answer --judge cascade
```

| Setting | Default | |
|---|---|---|
| `JEV_PROVIDER` | `auto` | `opencode` if `OPENCODE_API_KEY` is set, else `typesafe` (`TYPESAFE_API_KEY`, `jev-latest`) |
| `JEV_MODEL` | `jev-1.13-free` | set `jev-1.13` for the paid model |
| `JEV_FALLBACK_MODEL` | `jev-1.13` | used automatically if the free model is withdrawn (HTTP 402/403/404/410); `none` disables |
| `JEV_CASCADE_LOW` / `HIGH` | `0.3` / `0.7` | uncertainty band escalated to the LLM judge |

> ⚠️ The free model is offered "for a limited time" with no published end date, and the provider's no-training policy is only stated for paid usage. Don't send confidential data through `jev-1.13-free`; switch to `JEV_MODEL=jev-1.13` (≈ cents per thousand tests) for private systems.

Add your own checks:

```python
from agentic_qa import register_assertion, AssertionResult

@register_assertion
def cites_a_source(case):
    ok = "[source:" in case["sut_output"]
    return AssertionResult("cites_a_source", ok, severity="medium",
                           detail="" if ok else "Answer has no citation")
```

---

## 📓 Jupyter Notebook Usage

For interactive use with DataFrames and rendered reports, use the low-level API with the built-in `display_qa_results()` helper:

```python
import os
from agentic_qa.sut import CallableAdapter, set_active_sut
from agentic_qa.graph.workflow import run_qa_pipeline
from agentic_qa import display_qa_results

os.environ["MAX_ITERATIONS"] = "2"
os.environ["TESTS_PER_ITERATION"] = "3"

# Wrap your RAG function
adapter = CallableAdapter(
    fn=my_rag,
    system_name="My RAG App",
    description="A chatbot that answers questions about machine learning papers.",
    domain="Academic Research",
)
set_active_sut(adapter)

# Run the pipeline
final_state = run_qa_pipeline()

# Single call: renders report, metrics, and Pandas DataFrames
verdicts_df, regression_df = display_qa_results(final_state)
```

---

## 🏗️ Architecture

The framework is powered by **5 autonomous agents** orchestrated via a LangGraph state machine:

```
START
  │
  ▼
🔍 Discovery Agent
  │  Hypothesizes likely pipeline (unverified)
  ▼
🔴 Red-Team Agent ◄────────────────────┐
  │  Generates adversarial test cases   │
  ▼                                     │
⚡ Executor Agent                       │
  │  Runs tests concurrently w/ timeout │
  ▼                                     │
⚖️  Judge Agent                         │
  │  Assertions + policy → test review  │
  │  → LLM / Jev (+ optional RAGAS)     │
  ▼                                     │
 Decision ──── more iterations? ────────┘
  │
  ├── 🔧 Refiner Agent (analyzes failures, guides next round)
  │
  ▼
📊 Reporter Agent
  │  Compiles final Markdown QA report
  ▼
END
```

### Agent Roles

| Agent | Role |
|---|---|
| 🔍 **Discovery** | Asks an LLM to *hypothesize* the SUT's likely pipeline and weak points from its text description (black-box; nothing is inspected or verified). Used only as a hint for the Red-Team. |
| 🔴 **Red-Team** | Generates adversarial inputs (prompt injections, edge cases, boundary values, format violations), replays remembered failures, and writes at least one test per assigned policy rule each iteration. |
| ⚡ **Executor** | Runs test cases concurrently (per-test timeout) and captures the answer + retrieved contexts. |
| ⚖️ **Judge** | Layered evaluation: deterministic assertions and policy checks, a blind review of generated tests' expectations, LLM judge and/or Jev, optional RAGAS. Computes pass rate and coverage itself. |
| 🔧 **Refiner** | Analyzes failures, records only *new* patterns; when none are found the loop stops (convergence). |
| 📊 **Reporter** | Produces a comprehensive Markdown report with pass rates, top failures, missed facts, and fix recommendations. |

---

## 📊 What gets evaluated?

For each test case, the Judge Agent produces:

- ✅ / ❌ / 🚫 **Verdict** (`pass`, `fail`, `partial`, `error`, or `invalid` when the test itself was wrong) with reasoning
- 📜 **Policy violations**, including citations that don't exist in the retrieved contexts
- 🔒 **Assertion results** — and `model_status` when an assertion overrode the model
- 🎯 **Jev probabilities** + `needs_human_review` on judge disagreement (ensemble mode)
- 📈 **Local quality score** — keyword coverage, lexical similarity, length
- 🛡️ **Risk score** — injection detection, negative responses, crashes
- 🔎 **Retrieval inspection** — contexts found, relevance, noise ratio
- 🧠 **Hallucination detection** — grounded / unsupported / contradicted
- 🧮 **RAGAS metrics** (opt-in) — faithfulness, context precision, context recall, answer correctness, answer relevancy
- 🔁 **Regression memory** — tracks `first_failed_date`, `last_failed_date`, `recurrence_count`, `sut_version`

---

## 🖥️ Streamlit Dashboard

Run the included visual dashboard to monitor the pipeline in real time:

```bash
streamlit run app.py
```

Connect your own API endpoint, or use the built-in demo system to see the agents in action. Choose the judge backend in the sidebar, or switch to **Replay** mode and upload a saved run JSON to re-test the same suite.

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and fill in your keys:

```env
# Required
OPENAI_API_KEY=sk-...

# Models / judge
MODEL_NAME=gpt-4o-mini
JUDGE_MODEL_NAME=gpt-4o          # optional: stronger, independent judge
JUDGE_BACKEND=auto               # llm | jev | cascade | ensemble | auto
OPENCODE_API_KEY=...             # optional: free Jev (jev-1.13-free) via OpenCode Zen
# TYPESAFE_API_KEY=...           # or TypeSafe directly (paid)
JEV_PASS_THRESHOLD=0.5

# Judge tuning (defaults shown)
JUDGE_VALIDITY_MODEL=gpt-4o      # reviews generated tests' expectations
JUDGE_VALIDITY_CHECK=1           # 0 = skip the review
JUDGE_BATCH_SIZE=8               # tests per judge call
JUDGE_CONTEXT_CHARS=2000         # characters of each retrieved chunk shown to the judge

# Pipeline
MAX_ITERATIONS=3
TESTS_PER_ITERATION=5
POLICY_TESTS_PER_ITERATION=2     # policy rules the red team must attack per iteration
EXECUTOR_MAX_WORKERS=4
EXECUTOR_TIMEOUT=60
ENABLE_RAGAS=0
REGRESSION_MEMORY_REPLAY=3

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=agentic-qa
```

---

## 📁 Project Structure

```
agentic_qa/
├── agents/
│   ├── discovery.py        # Architecture hypothesis from the description (black-box)
│   ├── red_team.py         # Adversarial test generation
│   ├── executor.py         # SUT execution
│   ├── judge.py            # Layered judge, test-validity review, verdict reconciliation
│   ├── assertions.py       # Deterministic checks (+ register_assertion)
│   ├── jev_judge.py        # Jev judge: providers, free→paid fallback, cascade rule
│   ├── refiner.py          # Failure pattern analysis
│   ├── reporter.py         # Final report compilation
│   ├── local_evaluator.py  # Dependency-free local quality checks
│   └── regression_memory.py # Persistent failure tracking
├── cli.py                  # `agentic-qa run | diff | label | calibrate | rejudge | agreement | memory | bugs`
├── calibration.py          # Held-out judge calibration + `label` tool
├── golden.py               # Golden dataset loader, abstention detection, hallucination rate
├── benchmarks.py           # Planted-bug detectors for the demo SUT
├── policy.py               # policy.yaml → assertions, judge criteria, Jev questions, red-team targets
├── diff.py                 # Compare two runs (newly failing / fixed / metric deltas)
├── rejudge.py              # Re-judge saved runs; agreement with human labels
├── verdicts.py             # Verdict statuses (incl. `invalid`) shared across modules
├── taxonomy.py             # Attack taxonomy + measured coverage
├── graph/
│   ├── conditions.py       # Loop / stop routing (max iters, all-pass, convergence)
│   ├── workflow.py         # LangGraph state machine
│   └── state.py            # Shared QAState TypedDict
├── sut/
│   ├── base.py             # BaseSUTAdapter interface
│   ├── callable_adapter.py # Wraps Python functions
│   ├── api_adapter.py      # REST APIs (nested paths, body templates, env headers) + OpenAI-chat
│   ├── langchain_adapter.py # LangChain Runnables (retrieval chains, LCEL)
│   └── financial_rag.py    # Built-in demo SUT
├── schemas/
│   ├── test_case.py        # TestCase Pydantic model
│   └── verdict.py          # Verdict Pydantic model
└── utils/
    ├── llm.py              # Model factory (per-role models) + robust JSON parsing
    ├── run_io.py           # Run JSON export, summary metrics, replay loading
    ├── prompt_templates.py # All agent prompts
    └── summary.py          # display_qa_results() helper
examples/
├── research_paper_rag/     # LangChain RAG over a PDF: golden set + policy
└── youtube_rag/            # HTTP-tested YouTube assistant: golden set, citation policy, labels
tests/                      # Offline test suite (fake LLM — no API key needed)
```

---

## 🧪 Development

```bash
pip install -e ".[dev]"
pytest            # 120+ unit & end-to-end tests, fully offline (no API key)
```

CI runs the suite on Python 3.10–3.12 for every pull request and every push to `main`.

---

## 📦 Install options

```bash
pip install agentic-qa                 # core
pip install "agentic-qa[jev]"          # + Jev judge (free via OpenCode Zen)
pip install "agentic-qa[ragas]"        # + RAGAS metrics
pip install "agentic-qa[dashboard]"    # + Streamlit dashboard
```

---

## 📄 License

MIT — feel free to use, modify, and distribute.
