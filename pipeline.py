#pipeline.py -- generated from the bug-fixed research_system.ipynb
# Same logic as the notebook. Only mechanical changes for import-safety:
#  - no !pip install lines
#  - cell 15 (with_retry/log_event) moved before cell 14 (graph build),
#    since 14 calls with_retry() before 15 defines it in the notebook's own order
#  - final run-it cell wrapped in if __name__ == '__main__'

# ----- notebook cell 2 -----
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_tavily import TavilySearch
import requests
from langchain_groq import ChatGroq
import streamlit as st
import trafilatura
from typing import TypedDict, Annotated, Literal
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
import logging
import os
import time
import ipaddress
from urllib.parse import urlparse
from datetime import datetime, timezone
from langgraph.graph import StateGraph,START,END

# Logger must exist before any node/helper that might log — moved to the
# very top instead of living down near the graph-build cell (bug #5).
logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s")
logger=logging.getLogger("Research_Pipline")

def now_iso():
    # datetime.utcnow() is deprecated since Python 3.12 (bug #6).
    return datetime.now(timezone.utc).isoformat()

def is_safe_url(url: str) -> bool:
    """Basic SSRF guard (bug #18): only allow http(s) URLs that don't
    resolve to loopback/private/link-local addresses. Not exhaustive
    (doesn't cover DNS rebinding), but blocks the obvious cases before
    we hand an arbitrary URL to trafilatura.fetch_url."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        host = parsed.hostname
        if not host:
            return False
        if host in ("localhost",):
            return False
        try:
            ip = ipaddress.ip_address(host)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
        except ValueError:
            pass  # host is a domain name, not a literal IP — fine
        return True
    except Exception:
        return False

def upsert_citation(citation_store, *, title, url, source, snippet="", used_in=None):
    """Shared, deduped-by-URL citation & source-metadata store.
    Lives inside ResearchState['citation_store']."""
    existing = citation_store.get(url)
    if existing:
        if used_in and used_in not in existing.get("used_in", []):
            existing["used_in"].append(used_in)
    else:
        citation_store[url] = {
            "title": title,
            "url": url,
            "source": source,
            "snippet": snippet,
            "fetched_at": now_iso(),
            "used_in": [used_in] if used_in else [],
        }
    return citation_store

# ----- notebook cell 4 -----
load_dotenv()

def get_secret(key: str) -> str | None:
    """Check real env vars / .env first (local dev), then fall back to
    Streamlit Cloud's st.secrets (which does NOT auto-populate os.environ)."""
    val = os.getenv(key)
    if val:
        return val
    try:
        return st.secrets.get(key)
    except Exception:
        return None

GROQ_API_KEY=get_secret("GROQ_API_KEY")
TAVILY_API_KEY=get_secret("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in environment or st.secrets.")
if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY not found in environment or st.secrets.")

# Tavily's underlying wrapper reads TAVILY_API_KEY from the real environment
# rather than reliably honoring an api_key= kwarg — write it back explicitly
# so this works the same locally (.env) and on Streamlit Cloud (st.secrets).
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# ----- notebook cell 5 -----
# max_tokens bumped 1024 -> 2048 (bug #8): the writer/publisher structured
# calls (draft + citations + reference list + metadata) were tight against
# the old ceiling and risked silent truncation on longer drafts.
llm=ChatGroq(
   model="openai/gpt-oss-120b",
   temperature=0,
   max_tokens=2048
)
response=llm.invoke("Hello, are you working?")
print(response.content)

# ----- notebook cell 6 -----
class ResearchState(TypedDict):
  query:str
  search_results:list
  sources:list
  reader_outputs:list
  analysis:dict|None
  report:dict|None
  critic_feedback:dict|None
  final_report:dict|None
  citation_store:dict
  revision_count:int
  max_revisions:int
  errors:list
  events:list
  plan:dict|None
  searched_subtopics:list  # bug #14 — tracks subtopics already searched so retries don't re-burn Tavily quota

initial_state: ResearchState = {
    "query": "Impact of artificial intelligence on education",
    "search_results": [],
    "sources": [],
    "reader_outputs": [],
    "analysis": None,
    "report": None,
    "critic_feedback": None,
    "final_report": None,
    "citation_store": {},
    "revision_count": 0,
    "max_revisions": 2,
    "errors": [],
    "events": [],
    "plan": None,
    "searched_subtopics": []
}

print(initial_state)

# ============================================================
# GUARDRAILS — input, mid-pipeline, and output checks
# ============================================================

BANNED_TERMS = ["bomb", "weapon synthesis", "malware", "exploit code"]
# Word-boundary regex instead of plain substring match (bugs #9 / #19) —
# "bomb" as a bare `in` check would false-positive on "bombastic",
# "carpet bombing history", etc. This still isn't a robust safety filter
# (it's easy to evade with spacing/synonyms) but it removes the most
# obvious over-blocking false positives.
import re as _re
_BANNED_TERM_PATTERNS = [
    _re.compile(r"\b" + _re.escape(term) + r"\b") for term in BANNED_TERMS
]

def validate_query(query: str) -> tuple[bool, str]:
    q = query.strip()
    if len(q.split()) < 3:
        return False, "Query too short/vague — give a specific research topic."
    if len(q) > 500:
        return False, "Query too long — keep it under 500 characters."
    lowered = q.lower()
    for term, pattern in zip(BANNED_TERMS, _BANNED_TERM_PATTERNS):
        if pattern.search(lowered):
            return False, f"Query blocked — appears to request unsafe content ('{term}')."
    return True, ""

def validate_plan(plan: dict) -> tuple[bool, str]:
    if not plan or not plan.get("subtopics"):
        return False, "Planner produced no subtopics — cannot proceed to search."
    return True, ""

def validate_sources(sources: list) -> tuple[bool, str]:
    if not sources:
        return False, "Retriever found no sources — cannot proceed to reading."
    return True, ""

def validate_reader_outputs(reader_outputs: list) -> tuple[bool, str]:
    if not reader_outputs:
        return False, "Reader extracted nothing from any source — cannot analyze."
    return True, ""

def validate_output(critic_feedback: dict, final_report: dict | None = None) -> tuple[bool, str]:
    if not critic_feedback:
        return False, "No critic evaluation available — cannot verify report."
    if critic_feedback.get("hallucination_risk") == "high":
        return False, "Blocked: Critic flagged high hallucination risk."
    if critic_feedback.get("citation_check") == "poor":
        return False, "Blocked: Critic flagged poor citation quality."
    # Bug #2: this guardrail previously never checked that a final_report
    # was actually produced — a publisher failure that somehow slipped
    # past its own retry could have returned "ok" with nothing to show.
    if not final_report or not final_report.get("report"):
        return False, "Publisher produced no final report — cannot verify output."
    return True, ""


# ============================================================
# EVALUATION — quick automated metrics on a completed run
# ============================================================

def evaluate_run(output: dict) -> dict:
    final = output.get("final_report") or {}
    # Bug #20: the old denominator was every URL the *retriever* ever
    # found (`all_sources`), including ones the reader never processed
    # (capped at MAX_READER_SOURCES) and so could never have been cited —
    # that artificially deflates coverage. The real denominator is the
    # set of sources that actually reached the writer, i.e. anything the
    # reader processed (used_in contains "reader").
    all_sources = final.get("all_sources", [])
    citable_urls = {
        s.get("url") for s in all_sources
        if "reader" in (s.get("used_in") or [])
    }
    cited_urls = {r.get("url") for r in final.get("reference", []) if isinstance(r, dict)}
    coverage = len(cited_urls & citable_urls) / len(citable_urls) if citable_urls else 0

    scores = [e.get("score") for e in output.get("events", []) if e.get("node") == "critic" and "score" in e]

    return {
        "sources_gathered": len(citable_urls),
        "sources_cited": len(cited_urls),
        "citation_coverage": round(coverage, 2),
        "revision_count": output.get("revision_count", 0),
        "critic_score_history": scores,
        "final_critic_score": scores[-1] if scores else None,
        "errors_encountered": len(output.get("errors", [])),
    }

# ----- notebook cell 7 -----
class plannerOutPut(BaseModel):
    plan_id: str = Field(description="Short unique ID for this research plan")
    subtopics:list[str]=Field(min_length=2,
                    max_length=3,
                    description="2 focused research subtopics")
    strategy: str = Field(description="Brief research strategy" )

def planner_node(state:ResearchState)->ResearchState:
    prompt=ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a research planner.\n"
            "You MUST return the answer using the provided structured output schema.\n"
            "Do NOT answer conversationally.\n"
            "Do NOT ask the user for another query.\n"
            "Create exactly 2 focused research subtopics from the query.\n"
            "Return only the structured PlannerOutPut."
        ),
        (
            "human",
            "Research query: {query}\n"
            "Break this query into 2 or 3 focused subtopics."
        )
    ])
    Structured_llm=llm.with_structured_output(plannerOutPut,method="function_calling")
    chain=prompt|Structured_llm
    result=chain.invoke({"query":state["query"]})

    state["plan"]=result.model_dump()

    # Bug #4: guardrail moved here (fails fast at the node that produced
    # the bad data) instead of only being checked after the whole graph
    # finishes, which wastes retriever/reader/analyzer/writer/critic calls
    # on a run that was already doomed.
    plan_ok, plan_reason = validate_plan(state["plan"])
    if not plan_ok:
        raise ValueError(f"Guardrail: {plan_reason}")

    state["events"].append({"node":"planner","status":"success"})
    return state

# ----- notebook cell 8 -----
class SearchResultItem(BaseModel):
   title:str
   url:str
   snippet:str

class RetrieverOutPut(BaseModel):
   subtopic:str
   results:list[SearchResultItem]

_tavily_key_clean = (TAVILY_API_KEY or "").strip()
if _tavily_key_clean != TAVILY_API_KEY:
    TAVILY_API_KEY = _tavily_key_clean
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

Tavily=TavilySearch(max_results=2, api_key=TAVILY_API_KEY)

def retriever_node(state: ResearchState) -> ResearchState:
    subtopics = state["plan"]["subtopics"]

    # Bug #14: if with_retry has to re-run this node (e.g. one subtopic's
    # Tavily call raised), previously the whole node re-queried every
    # subtopic from scratch, burning Tavily quota on searches that already
    # succeeded. Now we carry forward what was already collected and skip
    # subtopics we've already searched successfully.
    all_results = list(state.get("search_results") or [])
    all_sources = list(state.get("sources") or [])
    searched_subtopics = set(state.get("searched_subtopics") or [])
    citation_store = state.setdefault("citation_store", {})

    # Maximum time allowed for collecting sources
    MAX_SEARCH_TIME = 10
    start_time = time.time()

    # One-time masked diagnostic
    _key = TAVILY_API_KEY or ""

    if len(_key) > 12:
        _mask = (
            f"{_key[:7]}...{_key[-4:]} "
            f"(len={len(_key)}, starts_with_tvly={_key.startswith('tvly-')})"
        )
    else:
        _mask = (
            f"SUSPICIOUSLY SHORT OR EMPTY: "
            f"'{_key}' (len={len(_key)})"
        )

    state["events"].append({
        "node": "retriever",
        "status": "key_diagnostic",
        "detail": _mask
    })

    for topic in subtopics:

        # Stop starting new searches after 10 seconds
        if time.time() - start_time >= MAX_SEARCH_TIME:
            state["events"].append({
                "node": "retriever",
                "status": "time_limit",
                "detail": "10-second search window reached"
            })
            break

        if topic in searched_subtopics:
            state["events"].append({
                "node": "retriever",
                "status": "subtopic_skip_cached",
                "detail": f"subtopic='{topic}' already searched in a prior attempt — reusing cached results instead of re-querying Tavily"
            })
            continue

        try:
            raw = Tavily.invoke({
                "query": topic
            })

        except Exception as e:
            err_detail = f"{type(e).__name__}: {e}"

            logger.error(
                f"[retriever] Tavily call failed "
                f"for subtopic '{topic}': {err_detail}"
            )

            state["errors"].append({
                "node": "retriever",
                "subtopic": topic,
                "error": err_detail
            })

            state["events"].append({
                "node": "retriever",
                "status": "subtopic_error",
                "detail": (
                    f"subtopic='{topic}' raised {err_detail}"
                )
            })

            continue

        raw_result = (
            raw.get("results", [])
            if isinstance(raw, dict)
            else raw
        )

        logger.info(
            f"[retriever] subtopic='{topic}' "
            f"-> {len(raw_result)} raw results"
        )

        if not raw_result:
            raw_preview = str(raw)[:300]

            logger.warning(
                f"[retriever] Tavily returned ZERO results "
                f"for subtopic '{topic}'. Raw response: {raw}"
            )

            state["events"].append({
                "node": "retriever",
                "status": "subtopic_empty",
                "detail": (
                    f"subtopic='{topic}' -> 0 results. "
                    f"raw={raw_preview}"
                )
            })

        else:
            state["events"].append({
                "node": "retriever",
                "status": "subtopic_ok",
                "detail": (
                    f"subtopic='{topic}' -> "
                    f"{len(raw_result)} results"
                )
            })

        for r in raw_result:

            item = {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get(
                    "content",
                    r.get("snippet", "")
                )
            }

            # Ignore results without a URL
            if not item["url"]:
                continue

            all_results.append(item)

            all_sources.append({
                "title": item["title"],
                "url": item["url"],
                 "snippet": item["snippet"]
            })
            

            upsert_citation(
                citation_store,
                title=item["title"],
                url=item["url"],
                source="web_search",
                snippet=item["snippet"],
                used_in="retriever"
            )

        # Mark this subtopic done (success or legitimately-empty) so a
        # retry of this node won't re-query it. Left unmarked on exception
        # (see `continue` in the except block above) so a transient
        # failure still gets retried for that specific subtopic.
        searched_subtopics.add(topic)

    # Calculate actual search time
    elapsed = round(time.time() - start_time, 2)

    # Fail only if absolutely no sources were collected
    if not all_sources:
        raise ValueError(
            "Retriever found no sources. "
            "Check TAVILY_API_KEY, Tavily quota/rate limits, "
            "or try a less narrow query."
        )

    state["search_results"] = all_results
    state["sources"] = all_sources
    state["searched_subtopics"] = list(searched_subtopics)

    state["events"].append({
        "node": "retriever",
        "status": "success",
        "detail": (
            f"Collected {len(all_sources)} sources "
            f"in {elapsed} seconds"
        )
    })

    return state

# ----- notebook cell 9 -----
class ReaderOutPut(BaseModel):
  url:str
  title:str
  key_points:list[str]
  summary:str

SCRAPE_TIMEOUT_SECONDS = 10  # bug #16 — trafilatura.fetch_url() has no
                             # built-in timeout and can hang indefinitely
                             # on a slow/unresponsive host.

def scrape_url(url: str) -> str:
  if not is_safe_url(url):  # bug #18
    raise ValueError(f"Refusing to fetch unsafe/non-public URL: {url}")
  try:
    resp = requests.get(
        url,
        timeout=SCRAPE_TIMEOUT_SECONDS,
        headers={"User-Agent": "Mozilla/5.0 (research-pipeline-bot)"},
    )
    resp.raise_for_status()
  except requests.exceptions.RequestException as e:
    raise ValueError(f"Could not fetch {url}: {e}")
  text = trafilatura.extract(resp.text)
  if not text:
    raise ValueError(f"No extractable text at {url}")
  return text

def reader_node(state: ResearchState) -> ResearchState:
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You read scraped page text and extract key points and a short summary. "
            "Do not invent facts not in the text."
            """Return only information directly supported by the provided sources.

            - Generate a maximum of 5 key points.
            - Each key point must be concise.
            - Each key point should be no more than 25 words.
            - Keep the summary below 80 words.
            - Do not add outside knowledge.
            - Do not speculate.
            - Do not repeat information. """
        ),
        (
            "human",
            "URL: {url}\nTitle: {title}\n\nText:\n{text}"
        )
    ])

    structured_llm = llm.with_structured_output(ReaderOutPut,method="function_calling")
    chain = prompt | structured_llm

    citation_store = state.setdefault("citation_store", {})
    reader_outputs = []

    # Bug #17: raised from 4000 now that max_tokens headroom (#8) can
    # actually support a longer per-source excerpt without truncating the
    # structured output. Still a hard cap by design — full-page text for
    # every source would blow the context budget across 3+ sources.
    MAX_READER_CHARS = 8000

    # Bug #7 (design tradeoff, not a pure bug): capping at 3 sources keeps
    # latency/cost bounded per run. Named as a constant so it's a
    # deliberate, easy-to-change knob instead of a silent magic number —
    # raise it if you want deeper coverage at the cost of more Groq calls.
    MAX_READER_SOURCES = 3

    for src in state["sources"][:MAX_READER_SOURCES]:
        try:
            text = scrape_url(src["url"])
        except Exception as e:
            state["errors"].append({"node": "reader", "url": src["url"], "error": str(e)})
            continue

        text = text.strip() if text else ""
        if len(text) < 50:
            state["errors"].append({"node": "reader", "url": src["url"], "error": f"scraped text too short ({len(text)} chars) — skipped"})
            continue

        if len(text) > MAX_READER_CHARS:
            text = text[:MAX_READER_CHARS]

        result = chain.invoke({
            "url": src["url"],
            "title": src["title"],
            "text": text
        })

        reader_outputs.append(result.model_dump())

        upsert_citation(
            citation_store,
            title=result.title or src["title"],
            url=result.url or src["url"],
            source="web",
            snippet=result.summary[:200],
            used_in="reader"
        )

    if not reader_outputs:
        raise ValueError(
            "Reader extracted no usable content from any source."
        )

    state["reader_outputs"] = reader_outputs

    state["events"].append({
        "node": "reader",
        "status": "success",
        "count": len(reader_outputs)
    })

    return state

# ----- notebook cell 10 -----
class AnalayerOutPut(BaseModel):
  findings:list[str]
  themes:list[str]
  insights: list[str]
  citation:list[str]

def analyzer_node(state:ResearchState)->ResearchState:
  if not state["reader_outputs"]:
    raise ValueError("Analyzer received no reader_outputs — nothing to synthesize or cite.")

  combined="\n\n".join(
        f"Source: {r['url']}\nTitle: {r['title']}\nSummary: {r['summary']}\nKey points: {r['key_points']}"
        for r in state["reader_outputs"]
    )

  # Guard against reader_outputs that exist but carry no real content
  # (e.g. every summary/key_points came back blank from the Reader LLM).
  substantive_chars = sum(
      len((r.get("summary") or "")) + len("".join(r.get("key_points") or []))
      for r in state["reader_outputs"]
  )
  if substantive_chars < 40:
    raise ValueError(
        f"Analyzer received {len(state.get('reader_outputs', []))} reader_outputs but they contain "
        f"almost no usable text (only {substantive_chars} chars of summary/key_points combined). "
        "The Reader LLM likely returned empty fields — check scraped page text quality "
        "and reader_node's structured output."
    )

  logger.info(f"[analyzer] combined context length: {len(combined)} chars from {len(state['reader_outputs'])} sources")

  prompt=ChatPromptTemplate.from_messages([
      (
        "system",
        "You are the Analyzer Agent. "
        "Analyze the research findings provided below. "
        "Identify important findings and useful insights. "
        "Use only information from the provided sources. "
        "Do not add outside facts or speculation. "
        "Keep the analysis concise."
    ),
    (
        "human",
        "Research Query: {query}\n\n"
        "Source Material:\n{combined}\n\n"
        "Reader Output:\n{reader_output}\n\n"
        "Return the required AnalayerOutPut structured output."
    )
  ])
  structured_llm=llm.with_structured_output(AnalayerOutPut,method="function_calling")
  chain=prompt|structured_llm

  result = chain.invoke({
    "query": state["query"],
    "combined": combined or "no sources available",
    "reader_output": "Reader outputs are already included in Source Material."
  })

  
  state["analysis"]=result.model_dump()
  state["events"].append({"node": "analyzer", "status": "success"})
  return state

# ----- notebook cell 11 -----
# ----- notebook cell 11: WRITER -----
class WriterOutPut(BaseModel):
    draft: str = Field(
        description="Concise final research report, maximum 400 words"
    )

    citation: list[str] = Field(
        description="Only URLs supplied in the source material"
    )


def writer_node(state: ResearchState) -> ResearchState:

    # -----------------------------
    # Get Analyzer output
    # -----------------------------

    analysis = state.get("analysis") or {}

    findings = analysis.get("findings") or []
    themes = analysis.get("themes") or []
    insights = analysis.get("insights") or []

    sources = state.get("sources") or []

    # -----------------------------
    # Guardrails
    # -----------------------------

    if not findings and not themes and not insights:
        raise ValueError(
            "Writer received no findings/themes/insights "
            "from Analyzer."
        )

    if not sources:
        raise ValueError(
            "Writer received no sources."
        )

    # -----------------------------
    # Revision feedback
    # -----------------------------

    revision_count = state.get("revision_count", 0)

    critic_feedback = state.get("critic_feedback") or {}

    feedback = ""

    if revision_count > 0:
        feedback = critic_feedback.get("feedback", "")

    # -----------------------------
    # Compact source material
    # -----------------------------

    reader_outputs = state.get("reader_outputs") or []

    source_blocks = []

    for i, reader in enumerate(reader_outputs[:3], start=1):

        url = reader.get("url", "")
        title = reader.get("title", "")
        summary = reader.get("summary", "")
        key_points = reader.get("key_points", [])

        source_blocks.append(
            f"""
SOURCE {i}
Title: {title}
URL: {url}
Summary: {summary}
Key Points: {key_points}
"""
        )

    combined_sources = "\n".join(source_blocks)

    # -----------------------------
    # Writer prompt
    # -----------------------------

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are the Writer Agent in a multi-agent research system.

Create the final research report using ONLY the
Analyzer Output and supplied Source Material.

STRICT RULES:

1. Use ONLY the provided information.
2. Do NOT use outside knowledge.
3. Do NOT invent facts, dates, statistics,
   events, people, organizations, or claims.
4. Keep the draft BELOW 350 words.
5. Use a clear title.
6. Use short sections.
7. Include a concise conclusion.
8. Stay directly focused on the user's research query.
9. citation must contain ONLY URLs that appear
   in the supplied Source Material.
10. Do NOT invent URLs.
11. Do NOT put URLs inside the draft.
12. Do NOT use [1], [2], [3], 【1】 or similar
    citation markers inside the draft.
13. If information is not supported by the sources,
    do not include it.
14. Apply Critic Feedback when provided.
15. Return ONLY the WriterOutPut structured output.
"""
        ),

        (
            "human",
            """
Research Query:
{query}

Analyzer Findings:
{findings}

Analyzer Themes:
{themes}

Analyzer Insights:
{insights}

Source Material:
{sources}

Critic Feedback:
{feedback}

Revision Number:
{revision}

Generate the final WriterOutPut.
"""
        )
    ])

    # -----------------------------
    # Structured output
    # -----------------------------

    structured_llm = llm.with_structured_output(
        WriterOutPut,
        method="function_calling"
    )

    chain = prompt | structured_llm

    # -----------------------------
    # Invoke Writer
    # -----------------------------

    result = chain.invoke({
        "query": state["query"],
        "findings": findings,
        "themes": themes,
        "insights": insights,
        "sources": combined_sources,
        "feedback": (
            feedback
            if feedback
            else "No critic feedback. This is the initial draft."
        ),
        "revision": revision_count
    })

    # -----------------------------
    # Validate citations
    # -----------------------------

    valid_urls = {
        source.get("url", "").strip()
        for source in sources
        if source.get("url")
    }

    invalid_citations = [
        url
        for url in result.citation
        if url not in valid_urls
    ]

    if invalid_citations:
        raise ValueError(
            "Writer generated citations that were not "
            f"present in supplied sources: {invalid_citations}"
        )

    # -----------------------------
    # Bug #11 (partial mitigation): existence != relevance. A full fix
    # needs a second LLM pass asking "does source X actually support
    # claim Y in the draft", which doubles writer-step cost/latency —
    # a real tradeoff worth deciding deliberately rather than silently
    # eating the extra Groq calls. As a free heuristic in the meantime,
    # flag (don't block) citations whose source content shares almost no
    # vocabulary with the draft — a cheap smell test, not a real semantic
    # check the critic's hallucination_risk pass (bug #1) is the actual
    # relevance backstop.
    # -----------------------------
    _STOPWORDS = {
        "the","a","an","of","in","on","and","or","to","for","is","are",
        "was","were","with","as","by","at","this","that","it","its","be",
        "from","has","have","had","not","but","which","also"
    }
    draft_words = {
        w.strip(".,;:()\"'").lower()
        for w in result.draft.split()
        if len(w) > 3 and w.strip(".,;:()\"'").lower() not in _STOPWORDS
    }
    for url in result.citation:
        source_info = next((s for s in sources if s.get("url") == url), {})
        source_text = f"{source_info.get('title','')} {source_info.get('snippet','')}"
        source_words = {
            w.strip(".,;:()\"'").lower()
            for w in source_text.split()
            if len(w) > 3 and w.strip(".,;:()\"'").lower() not in _STOPWORDS
        }
        if source_words and not (draft_words & source_words):
            state["events"].append({
                "node": "writer",
                "status": "citation_relevance_warning",
                "detail": f"'{url}' shares no vocabulary with the draft — possibly cited but not actually used"
            })

    # -----------------------------
    # Store report
    # -----------------------------

    state["report"] = result.model_dump()

    # -----------------------------
    # Update CitationStore
    # -----------------------------

    citation_store = state.setdefault(
        "citation_store",
        {}
    )

    for url in result.citation:

        source_info = next(
            (
                s for s in sources
                if s.get("url") == url
            ),
            {}
        )

        upsert_citation(
            citation_store,
            title=source_info.get(
                "title",
                "Writer citation"
            ),
            url=url,
            source="writer",
            snippet=source_info.get(
                "snippet",
                ""
            )[:200],
            used_in="writer"
        )

    # -----------------------------
    # Event logging
    # -----------------------------

    state["events"].append({
        "node": "writer",
        "status": "success",
        "revision": revision_count,
        "citation_count": len(result.citation)
    })

    return state

# ----- notebook cell 12 -----
class CriticOutPut(BaseModel):
    score: int = Field(
        description="Overall report quality score from 0 to 100"
    )

    feedback: str = Field(
        description="Concise feedback, maximum 150 words"
    )

    issues: list[str] = Field(
        description="Maximum 3 important issues"
    )

    suggestions: list[str] = Field(
        description="Maximum 3 specific improvements"
    )

    # Bug #10: Literal instead of free-form str — this makes the allowed
    # values part of the schema itself (enforced by structured output),
    # instead of relying on the prompt text and a downstream string match
    # that can silently drift out of sync (which is exactly how the old
    # "missing" vs "poor" mismatch happened).
    citation_check: Literal["good", "partial", "poor"] = Field(
        description="Citation quality: good, partial, or poor"
    )

    hallucination_risk: Literal["low", "medium", "high"] = Field(
        description=(
            "Risk that the draft contains claims NOT supported by the "
            "supplied citations/draft content. Must be exactly one of: "
            "low, medium, high."
        )
    )

def critic_node(state: ResearchState) -> ResearchState:

    draft = state.get("report", {}).get("draft", "")
    citation_urls = state.get("report", {}).get("citation", [])

    if not draft:
        raise ValueError("Critic received an empty draft.")

    # Bug #1: the critic previously only saw the bare citation URLs, with
    # no way to check whether the draft's claims are actually supported by
    # what those sources say. Pull the real content (title/summary/key
    # points) for each cited URL from citation_store so the critic can do
    # real fact-checking instead of guessing from a list of links.
    citation_store = state.get("citation_store", {})
    citation_blocks = []
    for url in citation_urls:
        entry = citation_store.get(url, {})
        citation_blocks.append(
            f"URL: {url}\n"
            f"Title: {entry.get('title', '(unknown)')}\n"
            f"Content: {entry.get('snippet', '(no content available)')}"
        )
    citations = "\n\n".join(citation_blocks) if citation_blocks else "No citations supplied."

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are the Critic Agent in a multi-agent research system.

Your job is to strictly evaluate the research report.

Evaluate the report using these criteria:

1. Accuracy — 30 points
2. Completeness — 25 points
3. Citation quality — 25 points
4. Relevance and clarity — 20 points

The final score must be between 0 and 100.

Rules:

- Evaluate ONLY the supplied draft and citations.
- Do NOT use outside knowledge.
- Do NOT invent missing facts.
- Check whether important claims are supported by the supplied citations.
- Identify specific weaknesses.
- Give practical improvements.
- Be strict. Do not automatically give a high score.
- Maximum 3 issues.
- Maximum 3 suggestions.
- Keep feedback concise.

Also assess hallucination_risk — whether the draft contains any claim,
fact, statistic, date, name, or event that is NOT directly supported by
the supplied draft/citations:
- "low": every claim is traceable to the supplied material.
- "medium": one minor unsupported detail.
- "high": multiple unsupported claims, or a fabricated citation/fact.
hallucination_risk must be exactly "low", "medium", or "high".

Return ONLY the CriticOutPut structured output.
"""
        ),
        (
            "human",
            """
Research Query:
{query}

Research Draft:
{draft}

Cited Sources (title + actual source content for each citation URL):
{citations}

Check each claim in the draft against the actual source content above —
not just whether the URL exists.

Now evaluate this research report.
"""
        )
    ])

    structured_llm = llm.with_structured_output(
        CriticOutPut,
        method="function_calling"
    )

    chain = prompt | structured_llm

    result = chain.invoke({
        "query": state["query"],
        "draft": draft,
        "citations": citations
    })

    if not 0 <= result.score <= 100:
        raise ValueError(
            f"Critic returned invalid score: {result.score}"
        )

    state["critic_feedback"] = result.model_dump()

    state["events"].append({
        "node": "critic",
        "status": "success",
        "score": result.score,
        "citation_check": result.citation_check,
        "hallucination_risk": result.hallucination_risk
    })

    return state
# ----- notebook cell 13 -----
class PublisherOutPut(BaseModel):
  report:str
  reference: list[dict]
  citation:list[str]
  metadata:dict

def publisher_node(state:ResearchState)->ResearchState:
  prompt=ChatPromptTemplate.from_messages([
      ("system",
         "You format an approved research draft into a clean final report "
         "with a title, clear sections, and a references list.\n\n"
         "STRICT RULES (bug #12 — publisher must not silently rewrite content):\n"
         "1. Formatting ONLY. Do not change facts, add new claims, add "
         "numbers/dates/names not already in the draft, or paraphrase in a "
         "way that changes meaning.\n"
         "2. `citation` and every `reference[].url` must come ONLY from the "
         "Sources list below. Do not invent, guess, or normalize a URL — "
         "copy it exactly as given.\n"
         "3. If you are not fully confident about a reference URL, omit it "
         "rather than guessing."),
        ("human", "Draft:\n{draft}\n\nSources (the ONLY URLs you may cite or reference):\n{sources}")
  ])

  structured_llm=llm.with_structured_output(PublisherOutPut,method="function_calling")
  chain=prompt|structured_llm

  result=chain.invoke({
      "draft": state["report"]["draft"],
        "sources": state["sources"]
  })

  # Bugs #3 / #13: previously nothing checked that the publisher's
  # citation list / reference URLs actually came from real, known
  # sources. An LLM can still normalize or slightly mangle a URL despite
  # instructions, so validate against citation_store (built from every
  # source the pipeline actually fetched) and fail the node — with_retry
  # will retry it — rather than silently shipping a fabricated reference.
  citation_store = state.get("citation_store", {})
  valid_urls = set(citation_store.keys())

  invalid_citations = [u for u in result.citation if u not in valid_urls]
  if invalid_citations:
      raise ValueError(
          f"Publisher citation URLs not found in citation_store: {invalid_citations}"
      )

  invalid_refs = [
      r.get("url") for r in result.reference
      if isinstance(r, dict) and r.get("url") and r.get("url") not in valid_urls
  ]
  if invalid_refs:
      raise ValueError(
          f"Publisher reference URLs not found in citation_store: {invalid_refs}"
      )

  final=result.model_dump()
  final["metadata"]={
      **final.get("metadata", {}),
      "critic_score": state["critic_feedback"]["score"],
      "critic_summary": state["critic_feedback"]["feedback"],
      "revision_count": state["revision_count"],
  }
  final["all_sources"]=list(state.get("citation_store", {}).values())

  state["final_report"]=final
  state["events"].append({"node": "publisher", "status": "success"})
  return state

# ----- notebook cell 15 -----
# (logger now defined near the top of the file, before it's ever used — bug #5)

def log_event(state:ResearchState,node:str,status:str,detail:str=""):
  entry={"node":node,"status":status,"detail":detail,"time":time.time()}
  state["events"].append(entry)
  if status=="error":
    logger.error(f"[{node}] {detail}")
  else:
    logger.info(f"[{node},{status},{detail}]")

def with_retry(node_name, max_attempts=1, backoff=1.5):
    def decorator(fn):
        def wrapper(state: ResearchState) -> ResearchState:
            attempt = 0
            while attempt < max_attempts:
                attempt += 1
                try:
                    log_event(state, node_name, "start", f"attempt {attempt}")
                    result = fn(state)
                    log_event(state, node_name, "success")
                    return result
                except Exception as e:
                    if attempt < max_attempts:
                        wait = backoff * attempt
                        if "rate_limit" in str(e) or "429" in str(e):
                            # TPM windows reset over seconds, not the
                            # ~20ms Groq suggests — give it real room.
                            wait = max(wait, 5.0 * attempt)
                        log_event(state, node_name, "retry", str(e))
                        time.sleep(wait)
                    else:
                        log_event(state, node_name, "error", str(e))
                        state["errors"].append({"node": node_name, "error": str(e)})
                        raise
        return wrapper
    return decorator

# ----- notebook cell 14 -----

 graph=StateGraph(ResearchState)
graph.add_node("planner",with_retry("planner",max_attempts=3,backoff=2.0)(planner_node))
graph.add_node("retriever",with_retry("retriever",max_attempts=2,backoff=2.0)(retriever_node))
graph.add_node("reader",with_retry("reader",max_attempts=3,backoff=2.0)(reader_node))
graph.add_node("analyzer",with_retry("analyzer",max_attempts=3,backoff=2.0)(analyzer_node))
graph.add_node("writer",with_retry("writer",max_attempts=3,backoff=2.0)(writer_node))
graph.add_node("critic",with_retry("critic",max_attempts=3,backoff=2.0)(critic_node))
graph.add_node("publisher",with_retry("publisher",max_attempts=3,backoff=2.0)(publisher_node))

graph.add_edge(START,"planner")
graph.add_edge("planner","retriever")
graph.add_edge("retriever","reader")
graph.add_edge("reader","analyzer")
graph.add_edge("analyzer","writer")
graph.add_edge("writer","critic")

def prepare_revision_node(state: ResearchState) -> ResearchState:
    """Increments revision_count. Must live in an actual graph node —
    conditional-edge routing functions cannot persist state mutations
    in LangGraph, they only decide where to go next."""
    state["revision_count"] = state.get("revision_count", 0) + 1
    state["events"].append({
        "node": "prepare_revision",
        "status": "success",
        "revision_count": state["revision_count"]
    })
    return state

graph.add_node("prepare_revision", with_retry("prepare_revision")(prepare_revision_node))
graph.add_edge("prepare_revision", "writer")

def route_after_critic(state: ResearchState) -> str:
    """Pure routing decision — reads state only, never mutates it."""

    critic_feedback = state.get("critic_feedback") or {}

    score = critic_feedback.get("score", 0)

    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 2)

    # Good enough → publish
    if score >= 75:
        return "publisher"

    # Maximum revisions already reached → publish
    if revision_count >= max_revisions:
        return "publisher"

    # Otherwise perform another revision
    return "prepare_revision"

graph.add_conditional_edges("critic",route_after_critic,
    {"prepare_revision": "prepare_revision", "publisher": "publisher"})
graph.add_edge("publisher",END)

app=graph.compile()

# ----- notebook cell 16 -----
class Orchestrator:
  def __init__(self,compiled_graph,max_revisions:int=2):
    self.app=compiled_graph
    self.max_revisions=max_revisions

  def build_initial_state(self,query:str)->ResearchState:
    return {
        "query": query,
        "search_results": [],
        "sources": [],
        "reader_outputs": [],
        "plan": None,
        "analysis": None,
        "report": None,
        "critic_feedback": None,
        "final_report": None,
        "citation_store": {},
        "revision_count": 0,
        "max_revisions": self.max_revisions,
        "errors": [],
        "events": [],
        "searched_subtopics": [],
    }

  def run(self,query:str)->dict:
       # --- input guardrail ---
       ok, reason = validate_query(query)
       if not ok:
           return {"ok": False, "error": f"Guardrail: {reason}", "errors": [], "events": []}

       state=self.build_initial_state(query)
       log_event(state, "orchestrator", "start", f"query={query}")

       try:
        result=self.app.invoke(state)

        # --- mid-pipeline guardrails ---
        # Bug #4: the primary enforcement now happens inside planner_node /
        # retriever_node / reader_node themselves (they raise and halt the
        # graph immediately on bad data — see with_retry's exception
        # handling). These are kept only as a defense-in-depth safety net
        # in case a node's return value is ever reshaped without updating
        # its internal check.
        checks = [
            validate_plan(result.get("plan")),
            validate_sources(result.get("sources")),
            validate_reader_outputs(result.get("reader_outputs")),
        ]
        for c_ok, c_reason in checks:
            if not c_ok:
                log_event(result, "orchestrator", "error", f"pipeline guardrail: {c_reason}")
                return {"ok": False, "error": f"Guardrail: {c_reason}",
                        "errors": result.get("errors", []), "events": result.get("events", [])}

        # --- output guardrail ---
        out_ok, out_reason = validate_output(result.get("critic_feedback"), result.get("final_report"))
        if not out_ok:
            log_event(result, "orchestrator", "error", f"output guardrail: {out_reason}")
            return {"ok": False, "error": f"Guardrail: {out_reason}",
                    "errors": result.get("errors", []), "events": result.get("events", [])}

        log_event(result,"orchestrator", "success",
                   f"revisions_used={result['revision_count']}")

        return {
            "ok": True,
            "final_report": result.get("final_report"),
            "revision_count": result.get("revision_count"),
            "errors": result.get("errors", []),
            "events": result.get("events", []),
        }
       except Exception as e:
           log_event(state, "orchestrator", "error", str(e))
           return {
                "ok": False,
                "error": str(e),
                "errors": state.get("errors", []),
                "events": state.get("events", []),
         }

# ----- notebook cell 17 (guarded) -----
if __name__ == '__main__':
    orchestrator = Orchestrator(app, max_revisions=2)
    output = orchestrator.run("Impact of artificial intelligence on education")
    
    if output["ok"]:
        print(output["final_report"]["report"])
        print("Revisions used:", output["revision_count"])
    else:
        print("Pipeline failed:", output["error"])
    
    print("\n--- EVENTS ---")
    for ev in output["events"]:
        print(ev)
    
