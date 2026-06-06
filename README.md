# 🛡️ Agentic QA — Autonomous Multi-Agent Testing for RAG & LLMs

[![PyPI](https://img.shields.io/pypi/v/agentic-qa?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/agentic-qa/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-1C3C3C?logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Agentic QA** is an open-source Python library that autonomously tests RAG pipelines and LLM applications — no manual test cases required.

It operates as a **self-improving red-team loop**: a team of LangGraph agents generates adversarial inputs, executes them against your system, evaluates the responses, and refines its attack strategy based on failures — all without human intervention.

> Built to solve the problem of manual QA in production RAG systems, where writing edge-case test suites is time-consuming and often incomplete.

---

## 🔍 How is this different from RAGAS or DeepEval?

| Feature | RAGAS / DeepEval | **Agentic QA** |
|---|---|---|
| Test generation | ❌ Manual (you write test cases) | ✅ Autonomous (LLM red-team) |
| Self-improvement | ❌ Static | ✅ Refines tests based on failures |
| Adversarial testing | ❌ No | ✅ Injection, edge cases, boundary values |
| Regression memory | ❌ No | ✅ Replays past failures across runs |
| Architecture discovery | ❌ No | ✅ Graphify maps your SUT's internals |
| RAGAS metrics | ✅ Yes | ✅ Integrated (faithfulness, context precision, etc.) |

---

## 🚀 Quick Start

### Installation

```bash
pip install agentic-qa
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
🔍 Discovery Agent (Graphify)
  │  Maps the SUT's internal architecture
  ▼
🔴 Red-Team Agent ◄────────────────────┐
  │  Generates adversarial test cases   │
  ▼                                     │
⚡ Executor Agent                       │
  │  Runs tests against the SUT         │
  ▼                                     │
⚖️  Judge Agent                         │
  │  Evaluates outputs (LLM-as-Judge)   │
  │  + RAGAS metrics + Local evaluator  │
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
| 🔍 **Discovery (Graphify)** | Deduces the internal graph/pipeline of the SUT from its description. |
| 🔴 **Red-Team** | Generates adversarial inputs: prompt injections, edge cases, boundary values, format violations. |
| ⚡ **Executor** | Runs each test case against the SUT and captures outputs + retrieved contexts. |
| ⚖️ **Judge** | Evaluates outputs with LLM-as-Judge + RAGAS metrics (faithfulness, context precision, recall, answer correctness, relevancy). |
| 🔧 **Refiner** | Analyzes failure patterns and instructs the Red-Team on what to target next. |
| 📊 **Reporter** | Produces a comprehensive Markdown report with pass rates, top failures, missed facts, and fix recommendations. |

---

## 📊 What gets evaluated?

For each test case, the Judge Agent produces:

- ✅ / ❌ **Pass/Fail verdict** with detailed reasoning
- 📈 **Local quality score** — keyword coverage, lexical similarity, length
- 🛡️ **Risk score** — injection detection, negative responses, crashes
- 🔎 **Retrieval inspection** — contexts found, relevance, noise ratio
- 🧠 **Hallucination detection** — grounded / unsupported / contradicted
- 🧮 **RAGAS metrics** — faithfulness, context precision, context recall, answer correctness, answer relevancy
- 🔁 **Regression memory** — tracks `first_failed_date`, `last_failed_date`, `recurrence_count`, `sut_version`

---

## 🖥️ Streamlit Dashboard

Run the included visual dashboard to monitor the pipeline in real time:

```bash
streamlit run app.py
```

Connect your own API endpoint, or use the built-in demo system to see the agents in action.

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and fill in your keys:

```env
# Required
OPENAI_API_KEY=sk-...

# Optional: controls pipeline behaviour
MODEL_NAME=gpt-4o-mini
MAX_ITERATIONS=3
TESTS_PER_ITERATION=5

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
│   ├── discovery.py        # Graphify — architecture deduction
│   ├── red_team.py         # Adversarial test generation
│   ├── executor.py         # SUT execution
│   ├── judge.py            # LLM-as-Judge + RAGAS evaluation
│   ├── refiner.py          # Failure pattern analysis
│   ├── reporter.py         # Final report compilation
│   ├── local_evaluator.py  # Dependency-free local quality checks
│   └── regression_memory.py # Persistent failure tracking
├── graph/
│   ├── workflow.py         # LangGraph state machine
│   └── state.py            # Shared QAState TypedDict
├── sut/
│   ├── base.py             # BaseSUTAdapter interface
│   ├── callable_adapter.py # Wraps Python functions
│   ├── api_adapter.py      # Wraps REST API endpoints
│   └── financial_rag.py    # Built-in demo SUT
├── schemas/
│   ├── test_case.py        # TestCase Pydantic model
│   └── verdict.py          # Verdict Pydantic model
└── utils/
    ├── prompt_templates.py # All agent prompts
    └── summary.py          # display_qa_results() helper
```

---

## 📦 PyPI

Available on PyPI: [`agentic-qa`](https://pypi.org/project/agentic-qa/)

```bash
pip install agentic-qa
```

---

## 📄 License

MIT — feel free to use, modify, and distribute.
