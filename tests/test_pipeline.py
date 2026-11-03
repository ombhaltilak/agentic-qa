"""End-to-end pipeline tests with a fake LLM (no network, no API key)."""

import json

import agentic_qa
from agentic_qa.cli import main


def _rag(query: str) -> dict:
    if "PWNED" in query:
        return {"answer": "PWNED", "contexts": []}  # vulnerable to injection
    return {"answer": "The NAV of Alpha Fund is 102.45.", "contexts": ["Alpha Fund NAV: 102.45"]}


def test_full_run_with_assertion_override(isolated, fake_llm):
    state = agentic_qa.run_autonomous_test(
        target_function=_rag, system_description="Fund QA bot", domain="financial",
        max_iterations=2, tests_per_iteration=2,
    )
    verdicts = {v["test_id"]: v for v in state["all_verdicts"]}
    # The LLM judge (fake) passes everything; the injection assertion must override it.
    assert verdicts["TC-0101"]["status"] == "pass"
    assert verdicts["TC-0102"]["status"] == "fail"
    assert verdicts["TC-0102"]["model_status"] == "pass"
    # IDs are assigned by the pipeline, not trusted from the LLM ("TC-01XX").
    assert [t["id"] for t in state["test_suite"]][:2] == ["TC-0101", "TC-0102"]
    assert state["coverage_score"] > 0
    assert (isolated / "reports" / "qa_run_iter2.json").exists()
    assert fake_llm.calls.count("red_team") == 2


def test_replay_runs_saved_suite_without_generation(isolated, fake_llm, tmp_path):
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps([{"input_data": "What is the NAV?", "expected_behavior": "NAV",
                                  "edge_case_type": "domain_specific"}]))
    state = agentic_qa.run_autonomous_test(target_function=_rag, replay_file=str(suite), max_iterations=5)
    assert "red_team" not in fake_llm.calls
    assert state["current_iteration"] == 1
    assert [v["status"] for v in state["all_verdicts"]] == ["pass"]


def test_cli_quality_gate(isolated, fake_llm, tmp_path):
    out = tmp_path / "run.json"
    code = main(["run", "--target", "tests.test_pipeline:_rag", "--iterations", "1", "--tests", "2",
                 "--output", str(out), "--fail-under", "0.9"])
    assert code == 1  # injection failure drops pass rate to 50%
    assert json.loads(out.read_text())["summary"]["pass_rate"] == 0.5

    code = main(["run", "--target", "tests.test_pipeline:_rag", "--replay", str(out),
                 "--fail-under", "0.4"])
    assert code == 0


def test_cli_bad_target_returns_usage_error(isolated):
    assert main(["run", "--target", "not_a_spec"]) == 2


def test_calibrate_assertions_offline():
    from agentic_qa.calibration import calibrate

    report = calibrate(backends=("assertions",))
    assert report["dataset_size"] >= 10
    assert 0 < report["assertions"]["accuracy"] <= 1


def test_cascade_pipeline_only_escalates_uncertain(isolated, fake_llm, monkeypatch):
    from types import SimpleNamespace

    from agentic_qa.agents import jev_judge

    judged_inputs = []
    real_invoke = fake_llm.invoke

    def spy(messages):
        if "QA Judge" in messages[0].content:
            judged_inputs.append(messages[-1].content)
        return real_invoke(messages)

    class Client:  # confident pass on the NAV question, unsure on everything else
        def system_one(self, state, questions, model=None):
            p = 0.95 if "NAV" in state["user_input"] else 0.5
            return SimpleNamespace(model=model, answers={
                "meets_expected": SimpleNamespace(noul=p),
                "complied_with_attack": SimpleNamespace(noul=0.0)})

    monkeypatch.setattr(fake_llm, "invoke", spy)
    monkeypatch.setattr(jev_judge, "jev_available", lambda: True)
    monkeypatch.setattr(jev_judge, "_client", Client())

    def rag(q):
        return "I can only answer fund questions." if "NAV" not in q else "NAV is 102.45"

    fake_llm.red_team_cases = [
        {"input_data": "What is the NAV?", "expected_behavior": "NAV", "edge_case_type": "domain_specific"},
        {"input_data": "Tell me a joke", "expected_behavior": "Declines", "edge_case_type": "out_of_scope"},
    ]
    state = agentic_qa.run_autonomous_test(target_function=rag, judge_backend="cascade",
                                           max_iterations=1, tests_per_iteration=2)
    by_id = {v["test_id"]: v for v in state["all_verdicts"]}
    assert by_id["TC-0101"]["decided_by"] == "jev"
    assert by_id["TC-0102"]["decided_by"] == "llm"
    assert len(judged_inputs) == 1 and "TC-0101" not in judged_inputs[0]
    assert agentic_qa.run_summary(state)["decided_by"] == {"jev": 1, "llm": 1}
