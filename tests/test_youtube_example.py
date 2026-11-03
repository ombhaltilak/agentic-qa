from examples.youtube_rag.index_video import fmt_time, normalize
from agentic_qa.golden import is_abstention
from pathlib import Path

from agentic_qa.policy import load_policy, policy_assertions

POLICY = Path(__file__).resolve().parent.parent / "examples" / "youtube_rag" / "policy.yaml"


def test_normalize_transcript_formats():
    rows = normalize([{"text": "hi\nthere", "start": 75.4}, {"text": " "}, {"text": "x", "time": "1:02:03"}])
    assert rows == [{"text": "hi there", "time": "1:15"}, {"text": "x", "time": "1:02:03"}]
    assert fmt_time(3725) == "1:02:05"


def test_youtube_policy():
    rules = load_policy(POLICY)
    res = lambda out: {a.name.split(":")[1]: a.passed for a in policy_assertions({"sut_output": out}, rules)}
    assert res("Loops come next [03:15 - 04:30].")["cites_timestamps"]
    assert not res("Loops come next.")["cites_timestamps"]
    assert not res("[NO_SOURCES] sorry")["no_internal_tags"]
    abstain = "I'm sorry, that information is not in the video."
    assert is_abstention(abstain) and "cites_timestamps" not in res(abstain)


def test_regression_memory_is_scoped_per_system(monkeypatch):
    from agentic_qa.agents.regression_memory import memory_path
    monkeypatch.delenv("REGRESSION_MEMORY_PATH", raising=False)
    monkeypatch.setenv("REGRESSION_MEMORY_SCOPE", "YouTube RAG assistant")
    a = memory_path()
    monkeypatch.setenv("REGRESSION_MEMORY_SCOPE", "Research paper RAG")
    assert a != memory_path() and a.name == "youtube_rag_assistant.json"
    monkeypatch.setenv("REGRESSION_MEMORY_PATH", "/tmp/x.json")
    assert str(memory_path()) == "/tmp/x.json"


CITE = r"\[(\d{1,2}:\d{2}(?::\d{2})?\s*-\s*\d{1,2}:\d{2}(?::\d{2})?)\]"


def test_citation_check_catches_invented_timestamps():
    rules = load_policy(POLICY)
    ctx = ["[2:33 - 5:54]: neurons ...", "[10:43 - 14:02]: weights ..."]
    res = lambda out, c=ctx: {a.name: a for a in policy_assertions({"sut_output": out, "retrieved_contexts": c}, rules)}
    assert res("784 neurons [02:33 - 05:54]. Biases [10:43 - 14:02].")["policy:citations_exist"].passed
    bad = res("Digits [00:00 - 00:30]. Neurons [2:33 - 5:54].")["policy:citations_exist"]
    assert not bad.passed and "00:00 - 00:30" in bad.detail and bad.hard
    # can't verify without contexts -> no assertion rather than a guess
    assert "policy:citations_exist" not in res("Digits [00:00 - 00:30].", [])


def test_citation_check_whole_token_match():
    from agentic_qa.policy import citation_assertion
    rule = {"id": "c", "severity": "high", "citations": CITE}
    assert not citation_assertion("[2:33 - 5:54]", ["[12:33 - 5:54]: x"], rule).passed


def test_rules_to_target_rotates_least_attacked_first():
    from agentic_qa.policy import rules_to_target
    rules = load_policy(POLICY)
    first = [r["id"] for r in rules_to_target(rules, [], 2)]
    assert first == ["refuses_harmful", "citations_exist"]  # critical first, then high in file order
    suite = [{"target_rule": rid} for rid in first]
    second = [r["id"] for r in rules_to_target(rules, suite, 2)]
    assert not set(first) & set(second)


def test_red_team_fills_skipped_policy_rules(isolated, fake_llm, monkeypatch):
    from agentic_qa.agents.red_team import red_team_node
    monkeypatch.setenv("POLICY_TESTS_PER_ITERATION", "2")
    rules = load_policy(POLICY)
    out = red_team_node({"current_iteration": 0, "policy": rules, "test_suite": [], "all_verdicts": []})
    targets = [tc.get("target_rule") for tc in out["test_suite"]]
    assert {"refuses_harmful", "citations_exist"} <= set(targets)
    ids = [tc["id"] for tc in out["test_suite"]]
    assert len(ids) == len(set(ids))


def test_red_team_drops_unknown_target_rule(isolated, monkeypatch):
    from agentic_qa.agents.red_team import _keep_valid_targets
    cases = [{"target_rule": "made_up"}, {"target_rule": "refuses_harmful"}]
    _keep_valid_targets(cases, load_policy(POLICY))
    assert "target_rule" not in cases[0] and cases[1]["target_rule"] == "refuses_harmful"


def test_judge_payload_keeps_more_context(monkeypatch):
    import json
    from agentic_qa.agents.judge import _judge_payload
    payload = json.loads(_judge_payload([{"test_id": "t", "retrieved_contexts": ["x" * 5000]}]))
    assert len(payload[0]["retrieved_contexts"][0]) == 2000
    monkeypatch.setenv("JUDGE_CONTEXT_CHARS", "300")
    assert len(json.loads(_judge_payload([{"test_id": "t", "retrieved_contexts": ["x" * 5000]}]))[0]["retrieved_contexts"][0]) == 300


def test_golden_counts_policy_only_failures_as_correct_answers():
    from agentic_qa.golden import golden_metrics
    suite = [{"id": "a", "golden": True}, {"id": "b", "golden": True}, {"id": "c", "golden": True}]
    cite_fail = {"name": "policy:citations_exist", "passed": False, "hard": True}
    verdicts = [
        {"test_id": "a", "status": "fail", "model_status": "pass", "sut_output": "28x28 [0:00 - 0:30]",
         "assertions": [cite_fail], "policy_violations": ["citations_exist"]},
        {"test_id": "b", "status": "fail", "sut_output": "wrong answer"},
        {"test_id": "c", "status": "fail", "model_status": "pass", "sut_output": "x",
         "assertions": [cite_fail], "policy_violations": ["citations_exist", "grounded_in_video"]},
    ]
    g = golden_metrics(suite, verdicts)
    assert g["accuracy"] == round(1 / 3, 4) and g["hallucinated_test_ids"] == ["b", "c"]
    assert g["correct_but_policy_violation_ids"] == ["a"] and g["passed_all_checks"] == 0


def test_llm_judge_batches_and_retries_skipped(monkeypatch):
    from types import SimpleNamespace
    import json, re
    from agentic_qa.agents import judge
    calls = []

    class Skippy:
        def invoke(self, messages):
            ids = re.findall(r'"test_id": "(T\d+)"', messages[-1].content)
            calls.append(ids)
            keep = ids if len(calls) > 2 else ids[:-1]  # first round drops the last test of each batch
            return SimpleNamespace(content=json.dumps({"verdicts": [
                {"test_id": i, "status": "pass", "reasoning": "ok", "severity": "info", "confidence": 0.9} for i in keep]}))

    monkeypatch.setattr(judge.llm_utils, "get_llm", lambda *a, **k: Skippy())
    monkeypatch.setenv("JUDGE_BATCH_SIZE", "3")
    combined = [{"test_id": f"T{i}", "input_data": "q", "sut_output": "a"} for i in range(5)]
    verdicts, _ = judge._run_llm_judge({}, combined)
    assert [v["test_id"] for v in verdicts] == [f"T{i}" for i in range(5)]
    assert [len(c) for c in calls] == [3, 2, 2]
