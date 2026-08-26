"""
AI Research Agent System — Multi-Agent LangGraph Pipeline (Streamlit app)
==========================================================================

Architecture (matches the reference diagram):
  Input Guardrails -> Planner -> Retriever -> Reader -> Analyzer -> Writer
  -> Critic -> (revise? loop back to Writer, max 1 revision) -> Publisher
  -> Output Guardrails -> Final Report

Why this is token-efficient (fixes the "Groq using too many tokens" issue):
  1. No agentic tool-calling loops. The original notebook wrapped web_search
     and scrape_url inside `create_agent(...)`, which makes the LLM re-plan
     and re-call tools turn by turn (many extra tokens). Here, Retriever and
     Reader call Tavily / the scraper DIRECTLY in plain Python — zero LLM
     tokens spent on search or scraping.
  2. Every LLM call has a hard `max_tokens` cap sized to the job (planner:
     ~180, analyzer: ~350, writer: ~700, critic: ~180).
  3. All scraped/search text is truncated (SCRAPE_CHAR_LIMIT /
     SNIPPET_CHAR_LIMIT) before it is ever placed in a prompt.
  4. Only 1 revision loop by default (MAX_REVISIONS) instead of unlimited.
  5. A single small, fast Groq model is used everywhere by default.

Setup
-----
    pip install streamlit langgraph langchain-groq tavily-python trafilatura beautifulsoup4 requests python-dotenv

Create a .env file (same folder) with:
    GROQ_API_KEY=your_key_here
    TAVILY_API_KEY=your_key_here

Run
---
    streamlit run app.py
"""

import os
import re
from typing import TypedDict, List, Dict, Optional

import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Optional heavy imports are wrapped so the UI can still render a helpful
# error message instead of crashing if a package is missing.
# ---------------------------------------------------------------------------
IMPORT_ERROR = None
try:
    from tavily import TavilyClient
    from langchain_groq import ChatGroq
    from langgraph.graph import StateGraph, END
except Exception as e:  # pragma: no cover
    IMPORT_ERROR = str(e)

try:
    import trafilatura
    HAS_TRAFILATURA = True
except Exception:
    HAS_TRAFILATURA = False
    import requests
    from bs4 import BeautifulSoup

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG — tune these to trade off cost vs. depth
# ---------------------------------------------------------------------------
MODEL_NAME = "llama-3.1-8b-instant"   # small + fast + cheap on Groq
MAX_SUBTOPICS = 2                     # fewer subtopics -> fewer search calls
RESULTS_PER_SUBTOPIC = 2
TOP_SOURCES_TO_READ = 3
SCRAPE_CHAR_LIMIT = 1200
SNIPPET_CHAR_LIMIT = 250
MAX_REVISIONS = 1
PASS_SCORE = 75
BANNED_TERMS = {"hack", "exploit", "bomb", "weapon", "malware"}


# ---------------------------------------------------------------------------
# GUARDRAILS
# ---------------------------------------------------------------------------
def input_guardrail(query: str):
    """Returns (is_valid, list_of_issues)."""
    issues = []
    if not query or not query.strip():
        return False, ["Please enter a research topic."]
    words = query.strip().split()
    if len(words) < 5:
        issues.append("Query must contain at least 5 words.")
    if len(query) > 500:
        issues.append("Query must be under 500 characters.")
    hit = [t for t in BANNED_TERMS if t in query.lower()]
    if hit:
        issues.append(f"Query contains a banned term: {', '.join(hit)}")
    return len(issues) == 0, issues


def output_guardrail(report: str, citations: List[str]) -> Dict[str, bool]:
    return {
        "Has citations": len(citations) > 0,
        "Has a sources/references section": "source" in report.lower() or "reference" in report.lower(),
        "Meets minimum length (250 chars)": len(report) >= 250,
        "No placeholder text": not re.search(r"\b(lorem ipsum|todo|xxx|fill in)\b", report, re.I),
    }


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------
class ResearchState(TypedDict):
    topic: str
    plan: List[str]
    sources: List[Dict]
    scraped: List[Dict]
    analysis: str
    report: str
    critic_score: int
    critic_feedback: str
    revision_count: int
    final_report: str
    citations: List[str]
    output_checks: Dict[str, bool]
    errors: List[str]
    events: List[str]


def init_state(topic: str) -> ResearchState:
    return ResearchState(
        topic=topic, plan=[], sources=[], scraped=[], analysis="", report="",
        critic_score=0, critic_feedback="", revision_count=0, final_report="",
        citations=[], output_checks={}, errors=[], events=[],
    )


# ---------------------------------------------------------------------------
# LLM CLIENTS — separate, tightly capped max_tokens per role
# ---------------------------------------------------------------------------
def _llm(max_tokens: int, temperature: float = 0.2):
    return ChatGroq(model=MODEL_NAME, temperature=temperature, max_tokens=max_tokens)


def get_llms():
    return {
        "planner": _llm(180, 0),
        "analyzer": _llm(350, 0.2),
        "writer": _llm(700, 0.4),
        "critic": _llm(180, 0),
    }


def get_tavily():
    key = os.getenv("TAVILY_API_KEY")
    return TavilyClient(api_key=key) if key else None


def scrape_url(url: str) -> str:
    """Direct scrape — no LLM tokens spent here at all."""
    try:
        if HAS_TRAFILATURA:
            downloaded = trafilatura.fetch_url(url)
            text = trafilatura.extract(downloaded) or ""
        else:
            resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
        return text[:SCRAPE_CHAR_LIMIT] if text else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------------
def make_planner_node(llms):
    def planner_node(state: ResearchState) -> ResearchState:
        state["events"].append("🧠 Planner: creating research plan")
        prompt = (
            f"Break this research topic into {MAX_SUBTOPICS} short, distinct web search "
            f"queries that together cover it well.\nTopic: {state['topic']}\n"
            "Reply with ONLY the queries, one per line. No numbering, no extra text."
        )
        try:
            resp = llms["planner"].invoke(prompt)
            lines = [l.strip(" -•\t") for l in resp.content.strip().split("\n") if l.strip()]
            state["plan"] = lines[:MAX_SUBTOPICS] or [state["topic"]]
        except Exception as e:
            state["errors"].append(f"Planner error: {e}")
            state["plan"] = [state["topic"]]
        return state
    return planner_node


def retriever_node(state: ResearchState) -> ResearchState:
    state["events"].append("🔍 Retriever: searching the web for each subtopic")
    tavily = get_tavily()
    sources = []
    if not tavily:
        state["errors"].append("Retriever error: TAVILY_API_KEY is missing.")
        state["sources"] = sources
        return state
    for sub in state["plan"]:
        try:
            res = tavily.search(query=sub, max_results=RESULTS_PER_SUBTOPIC)
            for r in res.get("results", []):
                sources.append({
                    "subtopic": sub,
                    "title": r.get("title", "Untitled"),
                    "url": r.get("url", ""),
                    "snippet": (r.get("content", "") or "")[:SNIPPET_CHAR_LIMIT],
                })
        except Exception as e:
            state["errors"].append(f"Retriever error on '{sub}': {e}")
    state["sources"] = sources
    return state


def reader_node(state: ResearchState) -> ResearchState:
    state["events"].append("📄 Reader: reading top sources")
    scraped = []
    for s in state["sources"][:TOP_SOURCES_TO_READ]:
        text = scrape_url(s["url"]) if s.get("url") else ""
        scraped.append({"title": s["title"], "url": s["url"], "text": text or s["snippet"]})
    state["scraped"] = scraped
    return state


def make_analyzer_node(llms):
    def analyzer_node(state: ResearchState) -> ResearchState:
        state["events"].append("📊 Analyzer: synthesizing key findings")
        material = "\n\n".join(
            f"[{i+1}] {s['title']}\n{s['text'][:600]}" for i, s in enumerate(state["scraped"])
        )[:3000]
        prompt = (
            "From the material below, list the 3-4 most important, non-redundant "
            "findings as short bullet points. Use only facts present in the material.\n\n"
            f"Topic: {state['topic']}\n\nMaterial:\n{material}"
        )
        try:
            resp = llms["analyzer"].invoke(prompt)
            state["analysis"] = resp.content.strip()
        except Exception as e:
            state["errors"].append(f"Analyzer error: {e}")
            state["analysis"] = "\n".join(f"- {s['text'][:150]}" for s in state["scraped"])
        return state
    return analyzer_node


def make_writer_node(llms):
    def writer_node(state: ResearchState) -> ResearchState:
        if state.get("critic_feedback"):
            state["revision_count"] += 1
        state["events"].append(f"✍️ Writer: drafting report (revision {state['revision_count']})")
        revision_note = f"\nAddress this feedback: {state['critic_feedback']}" if state.get("critic_feedback") else ""
        prompt = (
            f"Write a concise research report (250-400 words) on: {state['topic']}\n\n"
            f"Key findings:\n{state['analysis']}{revision_note}\n\n"
            "Use exactly these headers: Introduction, Key Findings, Conclusion, Sources. "
            "Be factual and tight, no filler. Under Sources, list source titles only (one per line)."
        )
        try:
            resp = llms["writer"].invoke(prompt)
            state["report"] = resp.content.strip()
        except Exception as e:
            state["errors"].append(f"Writer error: {e}")
        return state
    return writer_node


def make_critic_node(llms):
    def critic_node(state: ResearchState) -> ResearchState:
        state["events"].append("🧐 Critic: evaluating the report")
        prompt = (
            "Score the report 0-100 for accuracy, depth and clarity. Reply in EXACTLY "
            "two lines:\nScore: <number>\nFeedback: <one short sentence>\n\n"
            f"Report:\n{state['report'][:1500]}"
        )
        try:
            text = llms["critic"].invoke(prompt).content
            score_m = re.search(r"Score:\s*(\d+)", text)
            fb_m = re.search(r"Feedback:\s*(.+)", text)
            state["critic_score"] = int(score_m.group(1)) if score_m else 70
            state["critic_feedback"] = fb_m.group(1).strip() if fb_m else ""
        except Exception as e:
            state["errors"].append(f"Critic error: {e}")
            state["critic_score"] = 70
        return state
    return critic_node


def should_revise(state: ResearchState) -> str:
    if state["critic_score"] >= PASS_SCORE or state["revision_count"] >= MAX_REVISIONS:
        return "publisher"
    return "writer"


def publisher_node(state: ResearchState) -> ResearchState:
    state["events"].append("📢 Publisher: finalizing report")
    citations = [f"- {s['title']}: {s['url']}" for s in state["sources"] if s.get("url")]
    state["citations"] = citations
    final = state["report"]
    if "sources" not in final.lower() and citations:
        final += "\n\n## Sources\n" + "\n".join(citations)
    state["final_report"] = final
    state["output_checks"] = output_guardrail(final, citations)
    return state


# ---------------------------------------------------------------------------
# GRAPH
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_graph():
    llms = get_llms()
    graph = StateGraph(ResearchState)
    graph.add_node("planner", make_planner_node(llms))
    graph.add_node("retriever", retriever_node)
    graph.add_node("reader", reader_node)
    graph.add_node("analyzer", make_analyzer_node(llms))
    graph.add_node("writer", make_writer_node(llms))
    graph.add_node("critic", make_critic_node(llms))
    graph.add_node("publisher", publisher_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "reader")
    graph.add_edge("reader", "analyzer")
    graph.add_edge("analyzer", "writer")
    graph.add_edge("writer", "critic")
    graph.add_conditional_edges("critic", should_revise, {"writer": "writer", "publisher": "publisher"})
    graph.add_edge("publisher", END)
    return graph.compile()


class ResearchOrchestrator:
    """
    Single point of coordination for the whole pipeline.

    This is the explicit orchestrator referenced by the architecture diagram
    ("LangGraph Workflow / StateGraph Orchestration"). It owns the compiled
    graph, runs a topic through all 7 agents end-to-end, and is the only
    place that talks to the graph directly — the UI (or a CLI/script) never
    touches `graph.stream()` itself, it just calls `orchestrator.run(...)`.
    """

    def __init__(self):
        self.graph = build_graph()

    def run(self, topic: str, on_step=None) -> ResearchState:
        """
        Run the full pipeline for `topic`.
        `on_step(node_name, state)` is called after every node finishes,
        so a caller (e.g. the Streamlit UI) can show live progress.
        Returns the final ResearchState once the Publisher node completes.
        """
        state = init_state(topic.strip())
        final_state: ResearchState = state
        for update in self.graph.stream(state):
            node_name = list(update.keys())[0]
            final_state = update[node_name]
            if on_step:
                on_step(node_name, final_state)
        return final_state


@st.cache_resource(show_spinner=False)
def get_orchestrator() -> "ResearchOrchestrator":
    return ResearchOrchestrator()


# ---------------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Research Agent", page_icon="🔎", layout="centered")

st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #0b0e14 0%, #10131c 100%); }
h1, h2, h3 { color: #e8e8f0 !important; }
.pipeline-badge {
  display: inline-block; padding: 4px 10px; margin: 2px; border-radius: 999px;
  background: #1b2030; color: #9dd3ff; font-size: 12px; border: 1px solid #2b3350;
}
.report-box {
  background: #12151f; border: 1px solid #262c40; border-radius: 12px;
  padding: 22px 24px; margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🔎 AI Research Agent")
st.caption("Planner → Retriever → Reader → Analyzer → Writer → Critic → Publisher")
st.markdown(
    "".join(f'<span class="pipeline-badge">{n}</span>' for n in
            ["Planner", "Retriever", "Reader", "Analyzer", "Writer", "Critic", "Publisher"]),
    unsafe_allow_html=True,
)
st.write("")

if IMPORT_ERROR:
    st.error(f"Missing dependency: {IMPORT_ERROR}\n\nInstall requirements first (see top of app.py).")
    st.stop()

with st.sidebar:
    st.subheader("⚙️ Settings")
    st.write(f"**Model:** `{MODEL_NAME}`")
    st.write(f"**Max revisions:** {MAX_REVISIONS}")
    st.write(f"**Subtopics/query:** {MAX_SUBTOPICS}")
    st.write(f"**Pass score:** {PASS_SCORE}/100")
    if not os.getenv("GROQ_API_KEY"):
        st.warning("GROQ_API_KEY not set")
    if not os.getenv("TAVILY_API_KEY"):
        st.warning("TAVILY_API_KEY not set")
    st.caption("Token-saving design: no agent tool-loops, hard max_tokens caps, "
               "truncated inputs, single revision loop.")

topic = st.text_input(
    "Enter a research topic",
    placeholder="e.g. Impact of renewable energy adoption on global electricity grids",
)
run = st.button("Run Research", type="primary")

if run:
    ok, issues = input_guardrail(topic)
    if not ok:
        for i in issues:
            st.error(i)
    else:
        orchestrator = get_orchestrator()

        steps = ["planner", "retriever", "reader", "analyzer", "writer", "critic", "publisher"]
        step_pct = {s: int((i + 1) / len(steps) * 100) for i, s in enumerate(steps)}

        progress = st.progress(0, text="Starting...")
        log_box = st.empty()
        final_state = None

        def _on_step(node_name, state):
            progress.progress(min(step_pct.get(node_name, 100), 100), text=f"Running: {node_name}")
            log_box.markdown("\n".join(f"- {e}" for e in state.get("events", [])))

        try:
            final_state = orchestrator.run(topic, on_step=_on_step)
        except Exception as e:
            st.error(f"Pipeline failed: {e}")

        if final_state:
            progress.progress(100, text="Done")
            st.success("Research complete!")

            st.subheader("📄 Final Report")
            st.markdown(f'<div class="report-box">', unsafe_allow_html=True)
            st.markdown(final_state["final_report"])
            st.markdown("</div>", unsafe_allow_html=True)

            st.download_button(
                "⬇️ Download report (.md)",
                final_state["final_report"],
                file_name="research_report.md",
                mime="text/markdown",
            )

            with st.expander("📊 Output guardrails & evaluation"):
                for k, v in final_state.get("output_checks", {}).items():
                    st.write(("✅ " if v else "❌ ") + k)
                st.write(f"**Sources found:** {len(final_state['sources'])}")
                st.write(f"**Revisions used:** {final_state['revision_count']}/{MAX_REVISIONS}")
                st.write(f"**Final critic score:** {final_state['critic_score']}/100")
                st.write(f"**Errors:** {len(final_state['errors'])}")

            if final_state.get("errors"):
                with st.expander("⚠️ Errors"):
                    for e in final_state["errors"]:
                        st.write("- " + e)

            with st.expander("🧭 Plan & sources used"):
                st.write("**Plan:**")
                for p in final_state["plan"]:
                    st.write("- " + p)
                st.write("**Sources:**")
                for c in final_state["citations"]:
                    st.write(c)
