"""
Multi-Agent Autonomous QA System — Streamlit Dashboard.

A generic, adaptive dashboard that can test ANY RAG system.
Users can connect their RAG via API endpoint or use the built-in demo.
"""

import streamlit as st
import os
import json
from dotenv import load_dotenv

load_dotenv()

# ── Page Config ──
st.set_page_config(
    page_title="Multi-Agent Autonomous QA System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 { color: #fff; font-size: 2rem; font-weight: 800; margin-bottom: 0.3rem; }
    .main-header p { color: rgba(255,255,255,0.65); font-size: 1rem; font-weight: 300; }
    .agent-card {
        background: linear-gradient(145deg, rgba(30,30,50,0.9), rgba(20,20,40,0.95));
        border: 1px solid rgba(255,255,255,0.08); border-radius: 12px;
        padding: 1.2rem 1.5rem; margin-bottom: 1rem;
        transition: all 0.3s ease; box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    .agent-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 4px 25px rgba(99, 102, 241, 0.15); transform: translateY(-2px);
    }
    .agent-card h3 { color: #e2e8f0; font-weight: 600; margin-bottom: 0.5rem; font-size: 1.05rem; }
    .agent-card p { color: rgba(255,255,255,0.5); font-size: 0.85rem; line-height: 1.5; }
    .metric-card {
        background: linear-gradient(145deg, rgba(30,30,50,0.8), rgba(15,15,35,0.9));
        border: 1px solid rgba(255,255,255,0.06); border-radius: 12px;
        padding: 1.2rem 1.5rem; text-align: center;
    }
    .metric-value {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #a855f7);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-label {
        color: rgba(255,255,255,0.5); font-size: 0.8rem;
        text-transform: uppercase; letter-spacing: 1px; margin-top: 0.3rem;
    }
    .pipeline-flow {
        display: flex; align-items: center; justify-content: center;
        gap: 0.5rem; padding: 1rem; margin: 1rem 0; flex-wrap: wrap;
    }
    .pipeline-node {
        background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 8px; padding: 0.5rem 1rem;
        color: #a5b4fc; font-size: 0.85rem; font-weight: 500;
    }
    .pipeline-node.active {
        background: rgba(99, 102, 241, 0.4); border-color: #6366f1;
        color: #fff; box-shadow: 0 0 15px rgba(99, 102, 241, 0.3);
    }
    .pipeline-arrow { color: rgba(255,255,255,0.3); font-size: 1.2rem; }
    .sut-badge {
        background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(34,197,94,0.05));
        border: 1px solid rgba(34,197,94,0.3); border-radius: 10px;
        padding: 1rem 1.5rem; margin: 1rem 0;
    }
    .sut-badge h4 { color: #22c55e; margin-bottom: 0.3rem; }
    .sut-badge p { color: rgba(255,255,255,0.6); font-size: 0.85rem; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0c29, #1a1a3e); }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Multi-Agent Autonomous QA System</h1>
        <p>Plug in ANY RAG system — AI agents autonomously generate adversarial tests, execute them, and self-improve coverage</p>
    </div>
    """, unsafe_allow_html=True)


def render_pipeline_flow(active_node=None):
    nodes = [
        ("🔴", "Red-Team", "red_team"), ("⚡", "Executor", "executor"),
        ("⚖️", "Judge", "judge"), ("🔧", "Refiner", "refiner"), ("📊", "Reporter", "reporter"),
    ]
    html = '<div class="pipeline-flow">'
    for i, (icon, name, key) in enumerate(nodes):
        active_class = "active" if key == active_node else ""
        html += f'<div class="pipeline-node {active_class}">{icon} {name}</div>'
        if i < len(nodes) - 1:
            html += '<span class="pipeline-arrow">→</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_agent_cards():
    agents = [
        ("🔴 Red-Team", "Generates adversarial inputs targeting edge cases"),
        ("⚡ Executor", "Runs tests through the connected SUT"),
        ("⚖️ Judge", "LLM-as-Judge with pass/fail verdicts"),
        ("🔧 Refiner", "Analyzes failures, improves next iteration"),
        ("📊 Reporter", "Compiles comprehensive QA report"),
    ]
    cols = st.columns(len(agents))
    for i, (title, desc) in enumerate(agents):
        with cols[i]:
            st.markdown(f'<div class="agent-card"><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)


def render_metrics(state):
    total_tests = len(state.get("test_suite", []))
    all_verdicts = state.get("all_verdicts", [])
    pass_count = sum(1 for v in all_verdicts if v.get("status") == "pass")
    fail_count = sum(1 for v in all_verdicts if v.get("status") in ("fail", "error"))
    coverage = state.get("coverage_score", 0.0)
    iteration = state.get("current_iteration", 0)
    cols = st.columns(5)
    metrics = [
        (str(total_tests), "Tests Generated"), (str(pass_count), "Tests Passed"),
        (str(fail_count), "Tests Failed"), (f"{coverage:.0%}", "Coverage Score"),
        (str(iteration), "Iterations"),
    ]
    for col, (value, label) in zip(cols, metrics):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)


def render_verdicts_table(verdicts):
    if not verdicts:
        st.info("No verdicts yet. Run the pipeline to see results.")
        return
    import pandas as pd
    rows = []
    for v in verdicts:
        local = v.get("local_evaluation", {}) or {}
        retrieval = local.get("retrieval_inspection", {}) or {}
        hallucination = local.get("hallucination_detection", {}) or {}
        rows.append({
            "Test ID": v.get("test_id", "?"),
            "Status": v.get("status", "?").upper(),
            "Severity": v.get("severity", "?"),
            "Confidence": f"{v.get('confidence', 0):.0%}",
            "Local Quality": f"{local.get('local_quality_score', 0):.0%}" if local else "-",
            "Risk": f"{local.get('risk_score', 0):.0%}" if local else "-",
            "Contexts": retrieval.get("contexts_found", "-"),
            "Grounding": hallucination.get("verdict", "-"),
            "Security Flags": len(local.get("security_flags", [])) if local else 0,
            "Reasoning": v.get("reasoning", "")[:100] + "...",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("### 🔌 Connect Your RAG")

        sut_mode = st.radio(
            "System Under Test",
            ["🏠 Built-in Demo (Financial RAG)", "🌐 API Endpoint (Any RAG)", "📝 Description Only"],
            index=0,
            help="Choose how to connect your system"
        )

        api_url = ""
        input_key = "query"
        output_key = "output"
        sut_name = "Demo RAG System"
        sut_description = ""
        sut_domain = "general"

        if "API Endpoint" in sut_mode:
            st.markdown("##### API Configuration")
            api_url = st.text_input("Endpoint URL", placeholder="http://localhost:8000/chat")
            sut_name = st.text_input("System Name", value="My RAG System")
            sut_description = st.text_area(
                "Describe your RAG",
                placeholder="e.g., A YouTube video Q&A chatbot that answers questions about video content using transcript embeddings",
                height=80,
            )
            sut_domain = st.text_input("Domain", value="general", placeholder="e.g., education, healthcare, legal")
            input_key = st.text_input("Input JSON key", value="query", help="JSON key for the input in your API")
            output_key = st.text_input("Output JSON key", value="output", help="JSON key for the response from your API")

        elif "Description Only" in sut_mode:
            st.markdown("##### Describe Your System")
            st.caption("No API needed — agents will generate and evaluate tests based on your description alone.")
            sut_name = st.text_input("System Name", value="My System")
            sut_description = st.text_area(
                "Describe what your system does",
                placeholder="e.g., A medical chatbot that answers patient queries about symptoms and medications using a knowledge base of medical literature",
                height=100,
            )
            sut_domain = st.text_input("Domain", value="general")

        else:
            st.success("✅ Using built-in Demo RAG System")
            st.caption("A demo system with intentional weaknesses for the agents to discover.")

        st.markdown("---")
        st.markdown("### ⚙️ Pipeline Config")
        run_mode = st.radio("Run Mode", ["Autonomous (Agents generate tests)", "Benchmark (Fixed dataset)"], index=0)
        max_iter = st.slider("Max Iterations", 1, 10, int(os.getenv("MAX_ITERATIONS", "3")))
        tests_per_iter = st.slider("Tests per Iteration", 3, 15, int(os.getenv("TESTS_PER_ITERATION", "5")))
        model = st.text_input("LLM Model", value=os.getenv("MODEL_NAME", "gpt-4o-mini"))

        st.markdown("---")
        st.markdown("### 📡 LangSmith")
        tracing = os.getenv("LANGCHAIN_TRACING_V2", "false") == "true"
        langsmith_key = os.getenv("LANGCHAIN_API_KEY", "")
        if tracing and langsmith_key:
            st.success("✅ Tracing enabled")
            st.markdown("[Open LangSmith →](https://smith.langchain.com)")
        else:
            st.warning("⚠️ Not configured")

        return {
            "sut_mode": sut_mode, "api_url": api_url, "input_key": input_key,
            "output_key": output_key, "sut_name": sut_name, "sut_description": sut_description,
            "sut_domain": sut_domain, "max_iter": max_iter, "tests_per_iter": tests_per_iter, "model": model,
            "run_mode": run_mode,
        }


def setup_sut(config):
    """Configure the active SUT based on sidebar settings."""
    from agentic_qa.sut import set_active_sut, APIAdapter, CallableAdapter, FinancialDocumentRAG

    if "API Endpoint" in config["sut_mode"]:
        if not config["api_url"]:
            st.error("❌ Please enter your API endpoint URL")
            return False
        adapter = APIAdapter(
            endpoint=config["api_url"],
            description=config["sut_description"] or f"RAG system at {config['api_url']}",
            system_name=config["sut_name"],
            domain=config["sut_domain"],
            input_key=config["input_key"],
            output_key=config["output_key"],
        )
        set_active_sut(adapter)
        return True

    elif "Description Only" in config["sut_mode"]:
        if not config["sut_description"]:
            st.error("❌ Please describe your system")
            return False
        # Use a pass-through adapter that just returns the input
        # The agents will generate and evaluate based on description alone
        class DescriptionOnlyAdapter(CallableAdapter):
            def process(self, input_text):
                return {
                    "status": "success",
                    "output": f"[System would process: {input_text[:200]}]",
                    "note": "Description-only mode — no real SUT connected",
                }
        adapter = DescriptionOnlyAdapter(
            fn=lambda x: x,
            description=config["sut_description"],
            system_name=config["sut_name"],
            domain=config["sut_domain"],
        )
        set_active_sut(adapter)
        return True

    else:
        # Built-in demo
        set_active_sut(FinancialDocumentRAG())
        return True


def main():
    render_header()
    config = render_sidebar()
    render_pipeline_flow(active_node=st.session_state.get("active_node"))
    render_agent_cards()

    # Show active SUT info
    sut_mode = config["sut_mode"]
    if "API Endpoint" in sut_mode and config["api_url"]:
        st.markdown(f"""
        <div class="sut-badge">
            <h4>🌐 Connected: {config["sut_name"]}</h4>
            <p>{config["sut_description"] or config["api_url"]}</p>
        </div>
        """, unsafe_allow_html=True)
    elif "Description Only" in sut_mode and config["sut_description"]:
        st.markdown(f"""
        <div class="sut-badge">
            <h4>📝 Testing: {config["sut_name"]}</h4>
            <p>{config["sut_description"][:150]}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Run Pipeline ──
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_button = st.button("🚀 Launch Autonomous QA Pipeline", use_container_width=True, type="primary")

    if run_button:
        os.environ["MAX_ITERATIONS"] = str(config["max_iter"])
        os.environ["TESTS_PER_ITERATION"] = str(config["tests_per_iter"])
        os.environ["MODEL_NAME"] = config["model"]

        if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your-openai-api-key-here":
            st.error("❌ Please set OPENAI_API_KEY in your .env file")
            return

        if not setup_sut(config):
            return

        with st.spinner("🔄 Pipeline running... This may take a few minutes."):
            try:
                from agentic_qa.graph.workflow import build_qa_graph, get_initial_state

                graph = build_qa_graph()
                initial_state = get_initial_state()
                initial_state["max_iterations"] = config["max_iter"]
                if "Benchmark" in config["run_mode"]:
                    initial_state["benchmark_mode"] = True
                    # In a real app we'd load a CSV/JSON dataset here. For demo we mock it.
                    st.info("Running in Benchmark mode using fixed reference dataset.")
                    
                progress_bar = st.progress(0, text="Starting pipeline...")
                final_state = None

                for event in graph.stream(initial_state, stream_mode="values"):
                    final_state = event
                    current_iter = event.get("current_iteration", 0)
                    progress = min(current_iter / config["max_iter"], 1.0)

                    if event.get("final_report"):
                        active = "reporter"
                    elif event.get("judge_verdicts"):
                        active = "judge"
                    elif event.get("execution_results"):
                        active = "executor"
                    else:
                        active = "red_team"

                    st.session_state["active_node"] = active
                    progress_bar.progress(progress, text=f"Iteration {current_iter}/{config['max_iter']} — {active.replace('_', ' ').title()}")

                progress_bar.progress(1.0, text="✅ Pipeline complete!")
                st.session_state["final_state"] = final_state
                st.session_state["active_node"] = None

            except Exception as e:
                st.error(f"❌ Pipeline error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                return

    # ── Display Results ──
    final_state = st.session_state.get("final_state")

    if final_state:
        st.markdown("## 📊 Results Dashboard")
        render_metrics(final_state)
        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["📋 Verdicts", "🧪 Test Cases", "🔍 Failure Patterns", "📄 Report"])

        with tab1:
            st.markdown("### All Verdicts")
            render_verdicts_table(final_state.get("all_verdicts", []))

        with tab2:
            st.markdown("### Generated Test Cases")
            test_suite = final_state.get("test_suite", [])
            if test_suite:
                import pandas as pd
                tc_rows = [{
                    "ID": tc.get("id", "?"), "Type": tc.get("edge_case_type", "?"),
                    "Difficulty": tc.get("difficulty", "?"),
                    "Input": tc.get("input_data", "")[:80] + "...",
                    "Expected": tc.get("expected_behavior", "")[:60] + "...",
                } for tc in test_suite]
                st.dataframe(pd.DataFrame(tc_rows), use_container_width=True, hide_index=True)

        with tab3:
            st.markdown("### Identified Failure Patterns")
            patterns = final_state.get("failure_patterns", [])
            if patterns:
                for i, p in enumerate(patterns, 1):
                    st.markdown(f"**{i}.** {p}")
            else:
                st.success("No failure patterns identified!")
            pass_rates = final_state.get("iteration_pass_rates", [])
            if pass_rates:
                st.markdown("### Pass Rate by Iteration")
                import pandas as pd
                chart_df = pd.DataFrame({"Iteration": list(range(1, len(pass_rates) + 1)), "Pass Rate": [r * 100 for r in pass_rates]})
                st.line_chart(chart_df.set_index("Iteration"), y="Pass Rate")

        with tab4:
            st.markdown("### Final QA Report")
            report = final_state.get("final_report", "")
            if report:
                st.markdown(report)
            else:
                st.info("Report not yet generated.")
    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 2rem; color: rgba(255,255,255,0.4);">
            <p style="font-size: 3rem; margin-bottom: 1rem;">🚀</p>
            <p style="font-size: 1.1rem; font-weight: 500;">Connect your RAG system in the sidebar, then click "Launch"</p>
            <p style="font-size: 0.85rem;">Or use the built-in Financial RAG demo to see how it works</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
