"""
app.py — Streamlit UI for the multi-agent research pipeline.
Now surfaces guardrail blocks distinctly from crashes, and adds an
Evaluation tab showing evaluate_run() metrics.
"""

import os
import traceback

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Multi-Agent Research Assistant", layout="wide")
st.title("🔎 Multi-Agent Research Assistant")
st.caption("Planner → Retriever → Reader → Analyzer → Writer → Critic → Publisher, with guardrails and a revision loop.")

# --- load backend ---
backend_ok = True
backend_error = None
try:
    from pipeline import Orchestrator, app as compiled_graph, evaluate_run
except Exception:
    backend_ok = False
    backend_error = traceback.format_exc()

# --- sidebar ---
with st.sidebar:
    st.header("Settings")
    query = st.text_area("Research query", placeholder="e.g. The history and current state of Iran-United States relations", height=100)
    max_revisions = st.slider("Max revision cycles", 0, 3, 2)
    st.markdown("---")
    st.write("**Environment**")
    st.write(f"GROQ_API_KEY: {'✅ set' if os.environ.get('GROQ_API_KEY') else '❌ missing'}")
    st.write(f"TAVILY_API_KEY: {'✅ set' if os.environ.get('TAVILY_API_KEY') else '❌ missing'}")
    st.markdown("---")
    st.write("**Guardrails active**")
    st.caption("Input: length + banned-term check\nMid-pipeline: empty plan / sources / reader output\nOutput: hallucination risk + citation check")
    st.markdown("---")
    run_clicked = st.button("Run research", type="primary", disabled=not backend_ok)

if not backend_ok:
    st.error("pipeline.py failed to import. See the error below.")
    st.code(backend_error)
    st.stop()

if "output" not in st.session_state:
    st.session_state.output = None

if run_clicked:
    if not query.strip():
        st.warning("Enter a research query first.")
        st.stop()
    if not os.environ.get("GROQ_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
        st.error("Missing API keys. Set GROQ_API_KEY and TAVILY_API_KEY in your .env file.")
        st.stop()

    with st.spinner("Running the pipeline — search, read, analyze, write, critique…"):
        try:
            orchestrator = Orchestrator(compiled_graph, max_revisions=max_revisions)
            st.session_state.output = orchestrator.run(query)
        except Exception:
            st.session_state.output = {"ok": False, "error": traceback.format_exc(), "errors": [], "events": []}

output = st.session_state.output

if output is None:
    st.info("Enter a query in the sidebar and click **Run research** to begin.")
else:
    if not output.get("ok"):
        error_msg = output.get("error", "Unknown error")
        if error_msg.startswith("Guardrail:"):
            st.warning(f"🛡️ Blocked by guardrail — {error_msg.replace('Guardrail: ', '')}")
        else:
            st.error("Pipeline failed.")
            st.code(error_msg)
        if output.get("events"):
            with st.expander("Event log up to this point"):
                for ev in output["events"]:
                    st.write(ev)
        st.stop()

    final = output.get("final_report") or {}
    metrics = evaluate_run(output)

    tab_report, tab_citations, tab_critic, tab_eval, tab_log = st.tabs(
        ["📄 Report", "🔗 Citations & Sources", "🕵️ Critic", "📊 Evaluation", "🛠️ Execution Log"]
    )

    with tab_report:
        st.subheader("Final Report")
        st.write(final.get("report", "No report was generated."))
        st.caption(f"Revisions used: {output.get('revision_count', 0)} / {max_revisions}")

    with tab_citations:
        st.subheader("All Sources & Metadata")
        all_sources = final.get("all_sources", [])
        if not all_sources:
            st.write("No sources recorded.")
        else:
            for src in all_sources:
                with st.container(border=True):
                    st.markdown(f"**{src.get('title', 'Untitled source')}**")
                    st.write(src.get("url", ""))
                    st.caption(
                        f"source: {src.get('source', 'unknown')} · "
                        f"used in: {', '.join(src.get('used_in', []))} · "
                        f"fetched: {src.get('fetched_at', '')}"
                    )
                    if src.get("snippet"):
                        st.write(src["snippet"])

        st.subheader("References cited in the final report")
        refs = final.get("reference", [])
        if refs:
            for r in refs:
                st.write(f"- {r}")
        else:
            st.write("No references listed.")

    with tab_critic:
        st.subheader("Critic Evaluation")
        meta = final.get("metadata", {})
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Score", f"{meta.get('critic_score', '—')}/100")
        with col2:
            st.write(meta.get("critic_summary", "No critic summary recorded."))
        st.caption(f"Revision count: {meta.get('revision_count', 0)}")

    with tab_eval:
        st.subheader("Run Evaluation")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sources gathered", metrics["sources_gathered"])
        c2.metric("Sources cited", metrics["sources_cited"])
        c3.metric("Citation coverage", f"{metrics['citation_coverage']*100:.0f}%")
        c4.metric("Revisions used", metrics["revision_count"])

        c5, c6 = st.columns(2)
        c5.metric("Final critic score", metrics["final_critic_score"] if metrics["final_critic_score"] is not None else "—")
        c6.metric("Errors encountered", metrics["errors_encountered"])

        if metrics["critic_score_history"]:
            st.write("**Critic score across revision cycles:**")
            st.line_chart(metrics["critic_score_history"])

    with tab_log:
        st.subheader("Execution Log")
        events = output.get("events", [])
        if events:
            st.dataframe(events, use_container_width=True)
        else:
            st.write("No events recorded.")

        errors = output.get("errors", [])
        if errors:
            st.subheader("Errors recorded during the run")
            st.dataframe(errors, use_container_width=True)
