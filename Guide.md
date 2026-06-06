Here is the complete, structured Markdown file for the Multi-Agent Autonomous QA System. You can copy this directly to generate your project scaffolding.

```markdown
# Multi-Agent Autonomous QA System (LangGraph)

## 📌 System Overview
An autonomous, agentic pipeline built with LangGraph where AI agents collaborate to generate test cases, execute them, detect failures, and self-improve test coverage. Unlike traditional testing frameworks that measure outputs against static inputs, this system dynamically generates adversarial inputs to expose edge cases and hidden failure modes.

**Target System Under Test (SUT):** Financial Document Processing Pipeline / RAG (e.g., NAV extraction, fund statement parsing for WAM domains).

---

## 🏗️ Architecture & Testing Loop

The system operates in a continuous, autonomous loop until maximum iterations are reached or all tests pass.

`START` ➔ `Red-Team Agent` ➔ `Executor Agent` ➔ `Judge Agent` ➔ *Condition* ➔ `Refiner Agent` (Loop) OR `Reporter Agent` (End) ➔ `END`

### 🤖 Agent Responsibilities

* **Red-Team Agent:** Generates diverse, adversarial test inputs targeting edge cases, boundary conditions, and known failure modes in financial data extraction.
* **Executor Agent:** Runs the generated test inputs through the System Under Test (SUT) and collects the raw outputs.
* **Judge Agent:** Evaluates each output against expected behavior using an LLM-as-judge pattern, determining a `pass` or `fail` verdict along with reasoning.
* **Refiner Agent:** Analyzes failure patterns from the Judge's verdicts and uses these insights to generate improved, highly targeted new test cases for the next iteration.
* **Reporter Agent:** Compiles a final, structured QA report detailing coverage, failure rates, and identified system vulnerabilities.

---

## 🧠 LangGraph State Schema

The shared state passed between nodes in the graph ensures all agents have context of the current iteration, historical failures, and test results.

```python
from typing import TypedDict, List, Optional

class TestCase(TypedDict):
    id: str
    input_data: str
    expected_behavior: str
    edge_case_type: str

class TestResult(TypedDict):
    test_id: str
    sut_output: str
    execution_time: float

class Verdict(TypedDict):
    test_id: str
    status: str  # "pass" or "fail"
    reasoning: str

class QAState(TypedDict):
    sut_description: str
    domain: str  # e.g., "WAM financial documents"
    test_suite: List[TestCase]
    current_iteration: int
    max_iterations: int
    execution_results: List[TestResult]
    judge_verdicts: List[Verdict]
    failure_patterns: List[str]
    coverage_score: float
    final_report: Optional[str]

```

---

## 📂 Project Directory Structure

```text
multi_agent_qa/
├── agents/
│   ├── __init__.py
│   ├── red_team.py       # Adversarial test generation logic
│   ├── executor.py       # SUT interaction and output logging
│   ├── judge.py          # Evaluation and verdict generation
│   ├── refiner.py        # Failure analysis and test mutation
│   └── reporter.py       # Final summary and formatting
├── graph/
│   ├── __init__.py
│   ├── state.py          # QAState TypedDict definitions
│   ├── workflow.py       # LangGraph node connections and compilation
│   └── conditions.py     # Routing logic (iteration checks, failure checks)
├── sut/
│   ├── __init__.py
│   └── financial_rag.py  # The mocked or actual System Under Test
├── schemas/
│   ├── __init__.py
│   ├── test_case.py      # Pydantic models for validation
│   └── verdict.py        # Pydantic models for validation
├── utils/
│   ├── __init__.py
│   └── prompt_templates.py # Standardized prompts for each agent
├── app.py                # Main execution script / Streamlit UI
├── requirements.txt
└── README.md

```

---

## 🛠️ Tech Stack

* **Orchestration:** LangGraph / LangChain
* **LLM Engine:** Qwen 2.5 (or any local/cloud LLM capable of function calling)
* **Language:** Python 3.10+
* **Validation:** Pydantic
* **UI / Dashboard (Optional):** Streamlit or Rich (CLI) for visualizing the agentic loop

```

```