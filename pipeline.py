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
from datetime import datetime
from langgraph.graph import StateGraph,START,END

def now_iso():
    return datetime.utcnow().isoformat() + "Z"

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
llm=ChatGroq(
   model="openai/gpt-oss-120b",
   temperature=0,
   max_tokens=1024
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
    "plan": None
}

print(initial_state)

# ============================================================
# GUARDRAILS — input, mid-pipeline, and output checks
# ============================================================

BANNED_TERMS = ["bomb", "weapon synthesis", "malware", "exploit code"]

def validate_query(query: str) -> tuple[bool, str]:
    q = query.strip()
    if len(q.split()) < 3:
        return False, "Query too short/vague — give a specific research topic."
    if len(q) > 500:
        return False, "Query too long — keep it under 500 characters."
    lowered = q.lower()
    for term in BANNED_TERMS:
        if term in lowered:
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

def validate_output(critic_feedback: dict) -> tuple[bool, str]:
    if not critic_feedback:
        return False, "No critic evaluation available — cannot verify report."
    if critic_feedback.get("hallucination_risk") == "high":
        return False, "Blocked: Critic flagged high hallucination risk."
    if critic_feedback.get("citation_check") == "missing":
        return False, "Blocked: Critic flagged missing citations."
    return True, ""


# ============================================================
# EVALUATION — quick automated metrics on a completed run
# ============================================================

def evaluate_run(output: dict) -> dict:
    final = output.get("final_report") or {}
    store_urls = {s.get("url") for s in final.get("all_sources", [])}
    cited_urls = {r.get("url") for r in final.get("reference", []) if isinstance(r, dict)}
    coverage = len(cited_urls & store_urls) / len(store_urls) if store_urls else 0

    scores = [e.get("score") for e in output.get("events", []) if e.get("node") == "critic" and "score" in e]

    return {
        "sources_gathered": len(store_urls),
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

    state["events"].append({"node":"planner","status":"success"})
    state["plan"]=result.model_dump()
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

    all_results = []
    all_sources = []
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

def scrape_url(url:str)->str:
  downloaded=trafilatura.fetch_url(url)
  if downloaded is None:
    raise ValueError(f"Could not fetch {url}")
  text = trafilatura.extract(downloaded)
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

    MAX_READER_CHARS = 4000

    for src in state["sources"][:3]:
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

    citation_check: str = Field(
        description="Citation quality: good, partial, or poor"
    )

def critic_node(state: ResearchState) -> ResearchState:

    draft = state.get("report", {}).get("draft", "")
    citations = state.get("report", {}).get("citation", [])

    if not draft:
        raise ValueError("Critic received an empty draft.")

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

Citations:
{citations}

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
        "citation_check": result.citation_check
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
         "with a title, clear sections, and a references list. "
         "Do not change facts or add new claims — formatting only."),
        ("human", "Draft:\n{draft}\n\nSources:\n{sources}")
  ])

  structured_llm=llm.with_structured_output(PublisherOutPut,method="function_calling")
  chain=prompt|structured_llm

  result=chain.invoke({
      "draft": state["report"]["draft"],
        "sources": state["sources"]
  })

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
logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s")
logger=logging.getLogger("Research_Pipline")

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
                        log_event(state, node_name, "retry", str(e))
                        time.sleep(backoff * attempt)
                    else:
                        log_event(state, node_name, "error", str(e))
                        state["errors"].append({"node": node_name, "error": str(e)})
                        raise
        return wrapper
    return decorator

# ----- notebook cell 14 -----
graph=StateGraph(ResearchState)
graph.add_node("planner",with_retry("planner")(planner_node))
graph.add_node("retriever",with_retry("retriever")(retriever_node))
graph.add_node("reader",with_retry("reader")(reader_node))
graph.add_node("analyzer",with_retry("analyzer")(analyzer_node))
graph.add_node("writer",with_retry("writer")(writer_node))
graph.add_node("critic",with_retry("critic")(critic_node))
graph.add_node("publisher",with_retry("publisher")(publisher_node))

graph.add_edge(START,"planner")
graph.add_edge("planner","retriever")
graph.add_edge("retriever","reader")
graph.add_edge("reader","analyzer")
graph.add_edge("analyzer","writer")
graph.add_edge("writer","critic")

def route_after_critic(state: ResearchState) -> str:

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
    state["revision_count"] = revision_count + 1

    return "writer"

graph.add_conditional_edges("critic",route_after_critic,
    {"writer": "writer", "publisher": "publisher"})
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
        out_ok, out_reason = validate_output(result.get("critic_feedback"))
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
    
