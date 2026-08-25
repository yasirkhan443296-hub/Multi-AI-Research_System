# pipeline.py -- generated from the bug-fixed research_system.ipynb
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
from langchain_google_genai import ChatGoogleGenerativeAI

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
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")

# ----- notebook cell 5 -----
llm=ChatGoogleGenerativeAI(
   model="gemini-2.5-flash-lite",
   temperature=0,
   api_key=GEMINI_API_KEY,
    max_output_tokens=1024
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
  plan_id:str=Field(description="short unique id for this plan")
  subtopics:list[str]=Field(description="2 subtopics to research")
  strategy:str=Field(description="brief search strategy")

def planner_node(state:ResearchState)->ResearchState:
    prompt=ChatPromptTemplate.from_messages([
        ("system", "You break a research query into a clear plan with 3-5 subtopics."),
        ("human", "Query: {query}")
    ])
    Structured_llm=llm.with_structured_output(plannerOutPut)
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

Tavily=TavilySearch(max_results=5,api_key=TAVILY_API_KEY)

def retriever_node(state:ResearchState)->ResearchState:
  subtopics=state["plan"]["subtopics"]
  all_results=[]
  all_sources=[]
  citation_store=state.setdefault("citation_store",{})

  for topic in subtopics:
    raw=Tavily.invoke({"query":topic})
    raw_result=raw.get("results",[]) if isinstance(raw,dict) else raw

    for r in raw_result:
      item={
          "title": r.get("title", ""),
          "url": r.get("url", ""),
          "snippet": r.get("content", r.get("snippet", ""))
      }
      all_results.append(item)
      all_sources.append({"title":item["title"],"url":item["url"]})
      upsert_citation(citation_store, title=item["title"], url=item["url"],
                       source="web_search", snippet=item["snippet"], used_in="retriever")

  state["search_results"]=all_results
  state["sources"]=all_sources
  state["events"].append({"node":"retriever","status":"success"})
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

def reader_node(state:ResearchState)->ResearchState:
  prompt=ChatPromptTemplate.from_messages([
      ("system", "You read scraped page text and extract key points and a short summary. Do not invent facts not in the text."),
      ("human", "URL: {url}\nTitle: {title}\n\nText:\n{text}")
  ])
  structured_llm=llm.with_structured_output(ReaderOutPut)
  chain=prompt|structured_llm

  citation_store=state.setdefault("citation_store",{})
  reader_outputs=[]
  for src in state["sources"]:
    try:
      text=scrape_url(src["url"])
    except Exception as e:
        state["errors"].append({"node": "reader", "url": src["url"], "error": str(e)})
        continue

    result=chain.invoke({
        "url": src["url"],
        "title": src["title"],
        "text": text
    })
    reader_outputs.append(result.model_dump())
    upsert_citation(citation_store, title=result.title or src["title"], url=result.url or src["url"],
                     source="web", snippet=result.summary[:200], used_in="reader")

  state["reader_outputs"]=reader_outputs
  state["events"].append({"node": "reader", "status": "success", "count": len(reader_outputs)})
  return state

# ----- notebook cell 10 -----
class AnalayerOutPut(BaseModel):
  findings:list[str]
  themes:list[str]
  insights: list[str]
  citation:list[str]

def analyzer_node(state:ResearchState)->ResearchState:
  combined="\n\n".join(
        f"Source: {r['url']}\nTitle: {r['title']}\nSummary: {r['summary']}\nKey points: {r['key_points']}"
        for r in state["reader_outputs"]
    )
  prompt=ChatPromptTemplate.from_messages([
      ("system",
         "You synthesize research findings across multiple sources. "
         "Identify common themes, key findings, and insights. "
         "Every finding must be traceable to a source below — cite the URL."),
        ("human", "Query: {query}\n\nSource material:\n{combined}")
  ])
  structured_llm=llm.with_structured_output(AnalayerOutPut)
  chain=prompt|structured_llm

  result=chain.invoke({
      "query": state["query"],
        "combined": combined or "no sources available"
  })

  state["analysis"]=result.model_dump()
  state["events"].append({"node": "analyzer", "status": "success"})
  return state

# ----- notebook cell 11 -----
class WriterOutPut(BaseModel):
  draft:str
  citation:list[str]
  reference: list[dict]

def writer_node(state:ResearchState)->ResearchState:
  is_revision=state["revision_count"]>0
  feedback=state["critic_feedback"]["feedback"] if is_revision and state["critic_feedback"] else ""

  system=(
      "You write a clear, well-structured research report from the analysis below. "
      "Cite sources by URL. Do not include claims not supported by the findings."
  )
  if is_revision:
    system+=" This is a REVISION — directly address the critic feedback provided."

  prompt=ChatPromptTemplate.from_messages([
      ("system", system),
      ("human",
       "Query: {query}\n\n"
       "Findings: {findings}\nThemes: {themes}\nInsights: {insights}\n\n"
       "Available sources:\n{sources}\n\n"
       "Critic feedback (empty if first draft):\n{feedback}")
  ])
  structured_llm=llm.with_structured_output(WriterOutPut)
  chain=prompt|structured_llm

  result=chain.invoke({
      "query": state["query"],
      "findings": state["analysis"]["findings"],
      "themes": state["analysis"]["themes"],
      "insights": state["analysis"]["insights"],
      "sources": state["sources"],
      "feedback": feedback
  })

  state["report"]=result.model_dump()
  state["events"].append({
      "node": "writer",
      "status": "success",
      "revision": state["revision_count"]
  })
  return state

# ----- notebook cell 12 -----
class CriticOutPut(BaseModel):
  score:int=Field(ge=0,le=100)
  feedback:str
  issues:list[str]
  sugeest:list[str]
  citation_check:Literal["valid", "missing", "partial"]

def critic_node(state:ResearchState)->ResearchState:
  prompt=ChatPromptTemplate.from_messages([
      ("system",
         "You are a strict evaluator of research report drafts. "
         "Score 0-100 on accuracy, completeness, and citation quality. "
         "Be specific about what's missing or wrong."),
        ("human",
         "Query: {query}\n\nDraft:\n{draft}\n\nCitations used:\n{citations}")
  ])

  structured_llm=llm.with_structured_output(CriticOutPut)
  chain=prompt|structured_llm

  result=chain.invoke({
      "query": state["query"],
        "draft": state["report"]["draft"],
        "citations": state["report"].get("citation", [])
  })

  state["critic_feedback"]=result.model_dump()
  state["events"].append({
      "node": "critic",
        "status": "success",
        "score": result.score
  })

  return state

# ----- notebook cell 13 -----
class PublisherOutPut(BaseModel):
  report:str
  reference: list[dict]
  citation:list[str]
  metadata:dict

publisher_llm = llm.bind(max_tokens=600)
def publisher_node(state:ResearchState)->ResearchState:
  prompt=ChatPromptTemplate.from_messages([
      ("system",
         "You format an approved research draft into a SHORT final report. "
         "Hard limit: under 350 words total. Title + 3 short sections + a "
         "references list (title and url only, no extra commentary). "
         "Do not change facts or add new claims — formatting and trimming only."),
        ("human", "Draft:\n{draft}\n\nSources:\n{sources}")
  ])

  structured_llm=publisher_llm.with_structured_output(PublisherOutPut)
  chain=prompt|structured_llm

  result=chain.invoke({
      "draft": state["report"]["draft"][:2500],
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

def with_retry(node_name, max_attempts=3, backoff=1.5):
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
    score = state["critic_feedback"]["score"]
    if score >= 75 or state["revision_count"] >= state["max_revisions"]:
        return "publisher"
    state["revision_count"] += 1
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
    
