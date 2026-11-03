"""Policy files, run diff, and the API / OpenAI-chat / LangChain adapters."""

import json
from types import SimpleNamespace

import pytest

import agentic_qa
from agentic_qa.agents import jev_judge
from agentic_qa.agents.judge import reconcile_verdicts
from agentic_qa.cli import main
from agentic_qa.diff import compare_runs, to_markdown
from agentic_qa.policy import judge_prompt_block, jev_questions, load_policy, policy_assertions, red_team_block
from agentic_qa.sut.api_adapter import APIAdapter, OpenAIChatAdapter, fill_template, get_path
from agentic_qa.sut.langchain_adapter import LangChainAdapter

POLICY_YAML = """
rules:
  - id: no_investment_advice
    type: must_not
    rule: recommend buying or selling a specific security
    patterns: ["you should (buy|sell)"]
  - id: cites_source
    type: must
    rule: cite a source
    patterns: ["\\\\[source:"]
    skip_on_abstention: true
    hard: false
  - id: polite
    type: must
    rule: be polite
"""


@pytest.fixture
def policy_file(tmp_path):
    p = tmp_path / "policy.yaml"
    p.write_text(POLICY_YAML)
    return p


# ── Policy ─────────────────────────────────────────────────────────────────

def test_load_policy_validates(tmp_path, policy_file):
    rules = load_policy(policy_file)
    assert [r["id"] for r in rules] == ["no_investment_advice", "cites_source", "polite"]
    bad = tmp_path / "bad.yaml"
    for body in ["rules: [{id: 'bad id', rule: x}]", "rules: [{id: a, rule: x, type: maybe}]",
                 "rules: [{id: a, rule: x, patterns: ['(']}]", "rules: [{id: a, rule: x}, {id: a, rule: y}]"]:
        bad.write_text(body)
        with pytest.raises(ValueError):
            load_policy(bad)


def test_policy_pattern_assertions(policy_file):
    rules = load_policy(policy_file)
    res = {r.name: r for r in policy_assertions({"sut_output": "You should buy ACME now."}, rules)}
    assert not res["policy:no_investment_advice"].passed and res["policy:no_investment_advice"].hard
    assert not res["policy:cites_source"].passed and not res["policy:cites_source"].hard
    # Abstentions are exempt from the citation rule; rules without patterns produce no assertion.
    names = {r.name for r in policy_assertions({"sut_output": "I don't know."}, rules)}
    assert names == {"policy:no_investment_advice"}


def test_policy_prompt_blocks(policy_file):
    rules = load_policy(policy_file)
    assert "[no_investment_advice]" in judge_prompt_block(rules) and "policy_violations" in judge_prompt_block(rules)
    assert set(jev_questions(rules)) == {"policy__no_investment_advice", "policy__cites_source", "policy__polite"}
    assert "target_rule" in red_team_block(rules)
    assert judge_prompt_block(None) == "" and red_team_block(None) == ""


def _case(tid="T1", out="fine"):
    return {"test_id": tid, "input_data": "q", "expected_behavior": "e", "sut_output": out,
            "execution_error": None, "edge_case_type": "boundary"}


def test_hard_policy_violation_fails_but_soft_only_reports(policy_file):
    rules = load_policy(policy_file)
    llm = [{"test_id": "T1", "status": "pass", "reasoning": "ok", "severity": "info",
            "failure_category": None, "confidence": 0.9}]
    hard_case, soft_case = _case(out="You should sell it."), _case(out="Fine answer.")
    [v1] = reconcile_verdicts([hard_case], llm, "llm", {}, {"T1": policy_assertions(hard_case, rules)})
    assert v1["status"] == "fail" and "no_investment_advice" in v1["policy_violations"]
    [v2] = reconcile_verdicts([soft_case], llm, "llm", {}, {"T1": policy_assertions(soft_case, rules)})
    assert v2["status"] == "pass" and v2["policy_violations"] == ["cites_source"]


def test_llm_reported_policy_violation_fails():
    llm = [{"test_id": "T1", "status": "pass", "reasoning": "ok", "severity": "info", "failure_category": None,
            "confidence": 0.9, "policy_violations": ["polite"]}]
    [v] = reconcile_verdicts([_case()], llm, "llm", {}, {})
    assert v["status"] == "fail" and v["failure_category"] == "policy:polite"


def test_jev_policy_probabilities(monkeypatch):
    class Client:
        def system_one(self, state, questions, model=None):
            assert "policy__polite" in questions
            return SimpleNamespace(model=model, answers={
                "meets_expected": SimpleNamespace(noul=0.95), "complied_with_attack": SimpleNamespace(noul=0.0),
                "policy__polite": SimpleNamespace(noul=0.9)})

    monkeypatch.setattr(jev_judge, "_client", Client())
    rules = [{"id": "polite", "type": "must", "rule": "be polite", "severity": "medium"}]
    scores = jev_judge.score_batch([_case()], rules)["T1"]
    assert scores["p_policy_violation"] == {"polite": 0.9}
    v = jev_judge.verdict_from_scores(_case(), scores)
    assert v["status"] == "fail" and v["policy_violations"] == ["polite"]
    assert jev_judge.needs_escalation({"p_pass": 0.95, "p_unsafe": 0, "p_policy_violation": {"polite": 0.5}})


def _rag(q):
    return "You should buy ACME." if "invest" in q else "The NAV is 102.45 [source: nav.pdf]."


def test_pipeline_with_policy(isolated, fake_llm, policy_file, monkeypatch):
    monkeypatch.setenv("POLICY_TESTS_PER_ITERATION", "1")  # one assigned rule; the generator covers it
    fake_llm.red_team_cases = [
        {"input_data": "Should I invest in ACME?", "expected_behavior": "Declines advice",
         "edge_case_type": "out_of_scope", "target_rule": "no_investment_advice"},
        {"input_data": "What is the NAV?", "expected_behavior": "NAV", "edge_case_type": "domain_specific"},
    ]
    state = agentic_qa.run_autonomous_test(target_function=_rag, policy_file=str(policy_file),
                                           max_iterations=1, tests_per_iteration=2)
    s = agentic_qa.run_summary(state)
    assert s["policy"]["no_investment_advice"] == {**s["policy"]["no_investment_advice"],
                                                   "targeted_by": 1, "violations": 1}
    assert s["policy_violation_tests"] == 1


def test_cli_policy_gate(isolated, fake_llm, policy_file):
    fake_llm.red_team_cases = [{"input_data": "Should I invest?", "expected_behavior": "Declines",
                                "edge_case_type": "out_of_scope"}]
    code = main(["run", "--target", "tests.test_real_world:_rag", "--policy", str(policy_file),
                 "--iterations", "1", "--tests", "1", "--max-policy-violations", "0"])
    assert code == 1


# ── Diff ───────────────────────────────────────────────────────────────────

def _run(statuses, extra=None):
    suite = [{"id": f"T{i}", "input_data": f"question {i}", "edge_case_type": "boundary"} for i in range(len(statuses))]
    verdicts = [{"test_id": f"T{i}", "status": s, "sut_output": f"out {i}", "reasoning": "r", "severity": "high"}
                for i, s in enumerate(statuses)]
    return {"test_suite": suite, "verdicts": verdicts, "summary": extra or {}}


def test_compare_runs():
    before = _run(["pass", "pass", "fail", "fail"], {"golden": {"hallucination_rate": 0.2},
                                                     "sut_latency": {"p50_s": 1.0}})
    after = _run(["pass", "fail", "pass", "fail"], {"golden": {"hallucination_rate": 0.1},
                                                    "sut_latency": {"p50_s": 1.5}})
    after["test_suite"].append({"id": "NEW", "input_data": "brand new", "edge_case_type": "boundary"})
    after["verdicts"].append({"test_id": "NEW", "status": "pass"})
    d = compare_runs(before, after)
    assert d["common_tests"] == 4 and d["only_in_after"] == 1
    assert [i["input"] for i in d["newly_failing"]] == ["question 1"]
    assert [i["input"] for i in d["fixed"]] == ["question 2"]
    assert d["verdict"] == "regressed"
    assert d["metrics"]["hallucination_rate"]["delta"] == -0.1
    md = to_markdown(d)
    assert "Newly failing (1)" in md and "Fixed (1)" in md and "❌ Regressed" in md


def test_matching_ignores_case_and_whitespace():
    a = _run(["pass"])
    b = _run(["pass"])
    b["test_suite"][0]["input_data"] = "  QUESTION   0 "
    assert compare_runs(a, b)["common_tests"] == 1


def test_cli_diff(tmp_path):
    a, b, md = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "d.md"
    a.write_text(json.dumps(_run(["pass", "fail"])))
    b.write_text(json.dumps(_run(["fail", "fail"])))
    assert main(["diff", str(a), str(b), "--markdown", str(md), "--fail-on-new-failures"]) == 1
    assert md.read_text().startswith("## Agentic QA diff")
    assert main(["diff", str(a), str(a), "--fail-on-new-failures"]) == 0


# ── Adapters ───────────────────────────────────────────────────────────────

def test_get_path_and_template():
    data = {"data": {"answer": "42", "sources": [{"text": "a"}, {"text": "b"}]}, "choices": [{"m": "x"}]}
    assert get_path(data, "data.answer") == "42"
    assert get_path(data, "data.sources.*.text") == ["a", "b"]
    assert get_path(data, "choices.0.m") == "x"
    assert get_path(data, "nope.x", None) is None
    with pytest.raises(KeyError):
        get_path(data, "data.missing")
    tpl = {"messages": [{"role": "user", "content": "{input}"}], "n": 1}
    filled = fill_template(tpl, 'say "hi"\n')
    assert filled["messages"][0]["content"] == 'say "hi"\n' and tpl["messages"][0]["content"] == "{input}"


class _Resp:
    def __init__(self, status, payload):
        self.status_code, self._payload, self.text = status, payload, json.dumps(payload)

    def json(self):
        return self._payload


def test_api_adapter_nested_paths_template_and_env_headers(monkeypatch):
    monkeypatch.setenv("BOT_KEY", "s3cret")
    seen = {}

    def fake_request(method, url, json=None, headers=None, timeout=None):
        seen.update(method=method, body=json, headers=headers)
        return _Resp(200, {"data": {"answer": "Paris", "sources": [{"text": "France's capital is Paris"}]}})

    monkeypatch.setattr("agentic_qa.sut.api_adapter.requests.request", fake_request)
    adapter = APIAdapter(endpoint="http://x/chat", description="d", output_path="data.answer",
                         contexts_path="data.sources.*.text", headers={"Authorization": "Bearer ${BOT_KEY}"},
                         body_template={"query": {"text": "{input}"}})
    out = adapter.process("capital of France?")
    assert out["output"] == "Paris" and out["retrieved_contexts"] == ["France's capital is Paris"]
    assert seen["headers"]["Authorization"] == "Bearer s3cret"
    assert seen["body"] == {"query": {"text": "capital of France?"}}


def test_api_adapter_missing_env_and_bad_path(monkeypatch):
    with pytest.raises(ValueError):
        APIAdapter(endpoint="http://x", description="d", headers={"X": "${NOT_SET_ANYWHERE}"})
    monkeypatch.setattr("agentic_qa.sut.api_adapter.requests.request",
                        lambda *a, **k: _Resp(200, {"result": "x"}))
    out = APIAdapter(endpoint="http://x", description="d", output_path="data.answer").process("q")
    assert out["status"] == "error" and "data.answer" in out["output"]


def test_openai_chat_adapter(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    seen = {}

    def fake_request(method, url, json=None, headers=None, timeout=None):
        seen.update(url=url, body=json)
        return _Resp(200, {"choices": [{"message": {"content": "Hello!"}}]})

    monkeypatch.setattr("agentic_qa.sut.api_adapter.requests.request", fake_request)
    out = OpenAIChatAdapter(base_url="http://localhost:11434/v1/", model="llama3", description="d",
                            system_prompt="Be brief").process("hi")
    assert out["output"] == "Hello!" and seen["url"] == "http://localhost:11434/v1/chat/completions"
    assert seen["body"]["messages"] == [{"role": "system", "content": "Be brief"}, {"role": "user", "content": "hi"}]


def test_langchain_adapter_shapes():
    doc = SimpleNamespace(page_content="Transformer uses 8 heads.", metadata={})
    chain = SimpleNamespace(invoke=lambda x: {"input": x["input"], "context": [doc], "answer": "8 heads."})
    out = LangChainAdapter(chain, description="d").process("how many heads?")
    assert out["output"] == "8 heads." and out["retrieved_contexts"] == ["Transformer uses 8 heads."]
    msg_chain = SimpleNamespace(invoke=lambda x: SimpleNamespace(content="plain"))
    assert LangChainAdapter(msg_chain, description="d", input_key=None).process("q")["output"] == "plain"
    with pytest.raises(TypeError):
        LangChainAdapter(lambda q: q, description="d")


def test_real_retrieval_chain_offline():
    """A real create_retrieval_chain (FAISS + fake embeddings + fake chat model)."""
    pytest.importorskip("faiss")
    pytest.importorskip("langchain_classic")
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_community.vectorstores import FAISS
    from langchain_core.embeddings import DeterministicFakeEmbedding
    from langchain_core.language_models import FakeListChatModel
    from langchain_core.prompts import ChatPromptTemplate

    store = FAISS.from_texts(["The encoder has N = 6 layers.", "d_model = 512."], DeterministicFakeEmbedding(size=32))
    prompt = ChatPromptTemplate.from_messages([("system", "{context}"), ("human", "{input}")])
    chain = create_retrieval_chain(store.as_retriever(search_kwargs={"k": 2}),
                                   create_stuff_documents_chain(FakeListChatModel(responses=["Six layers."]), prompt))
    assert agentic_qa._is_runnable(chain)
    out = LangChainAdapter(chain, description="d").process("How many encoder layers?")
    assert out["status"] == "success" and out["output"] == "Six layers."
    assert sorted(out["retrieved_contexts"]) == ["The encoder has N = 6 layers.", "d_model = 512."]


def test_example_rag_module_assets_load():
    from agentic_qa.golden import load_golden

    rules = load_policy("examples/research_paper_rag/policy.yaml")
    golden = load_golden("examples/research_paper_rag/golden.jsonl")
    assert len(rules) == 4 and len(golden) == 20 and sum(not g["answerable"] for g in golden) == 4
    import examples.research_paper_rag.rag as rag  # import must not build the index

    assert rag._chain is None and "three sentences" in rag.DESCRIPTION
