<div align="center">

# 🔎 Multi-Agent AI Research System

### An End-to-End Agentic AI Research Pipeline Built with LangGraph

<p>
  <b>Plan → Search → Read → Analyze → Write → Critique → Revise → Publish</b>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/LangGraph-Orchestration-purple?style=for-the-badge">
  <img src="https://img.shields.io/badge/Groq-LLM-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Tavily-Web%20Search-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Streamlit-UI-red?style=for-the-badge&logo=streamlit">
</p>

<p>
  A multi-agent research system that transforms a natural-language research question
  into a structured, source-backed report using specialized workflow nodes,
  shared state, web search, citation tracking, quality evaluation, and iterative revision.
</p>

</div>

---

## 📌 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [💡 Problem](#-problem)
- [✨ What the System Does](#-what-the-system-does)
- [🏗️ Architecture](#️-architecture)
- [🔄 End-to-End Workflow](#-end-to-end-workflow)
- [🧠 Agents and Nodes](#-agents-and-nodes)
- [🗃️ Research State](#️-research-state)
- [🔗 Citation Store](#-citation-store)
- [🔁 Critic and Revision Loop](#-critic-and-revision-loop)
- [🛡️ Retry and Error Handling](#️-retry-and-error-handling)
- [🖥️ Streamlit Application](#️-streamlit-application)
- [🌐 HTML Demonstration](#-html-demonstration)
- [📁 Project Structure](#-project-structure)
- [⚙️ Installation](#️-installation)
- [🔐 Environment Variables](#-environment-variables)
- [▶️ Run the Project](#️-run-the-project)
- [🧪 Example](#-example)
- [🧩 Core Code](#-core-code)
- [📊 Output](#-output)
- [🚀 Future Improvements](#-future-improvements)
- [⚠️ Current Limitations](#️-current-limitations)
- [👨‍💻 Project Goal](#-project-goal)

---

# 🎯 Project Overview

The **Multi-Agent AI Research System** is an Agentic AI application designed to automate a complete research workflow.

Instead of asking one LLM to search, read, reason, write, and evaluate everything in a single call, the system separates the work into specialized stages.

```text
User Research Question
        │
        ▼
     Planner
        │
        ▼
    Retriever
        │
        ▼
      Reader
        │
        ▼
     Analyzer
        │
        ▼
      Writer
        │
        ▼
      Critic
        │
        ├───────────────┐
        │               │
    Score ≥ 75       Score < 75
        │               │
        ▼               ▼
    Publisher         Writer
        │               │
        ▼               └──────► Critic
  Final Research
      Report
```

The workflow is orchestrated with **LangGraph** and uses a shared `ResearchState` to pass information between nodes.

The uploaded implementation defines the complete state with query, search results, sources, reader outputs, analysis, report, critic feedback, final report, citations, revision counters, errors, events, and plan information. fileciteturn0file0L63-L75

---

# 💡 Problem

Traditional LLM applications often follow this pattern:

```text
Question → LLM → Answer
```

That approach can be insufficient for serious research because the model has to perform many different jobs at once.

This project separates those responsibilities:

| Problem | System Component |
|---|---|
| How should the question be researched? | Planner |
| Where can information be found? | Retriever |
| What does each source actually say? | Reader |
| What do the sources collectively tell us? | Analyzer |
| How should the research be written? | Writer |
| Is the report good enough? | Critic |
| How should the approved report be formatted? | Publisher |

This creates a more controlled and observable research pipeline.

---

# ✨ What the System Does

Given a query such as:

```text
Impact of artificial intelligence on education
```

the system performs:

### 1. Planning

The Planner breaks the question into 3–5 research subtopics.

### 2. Web Search

The Retriever searches those subtopics through Tavily.

### 3. Source Reading

The Reader fetches webpages and extracts their content.

### 4. Information Extraction

The Reader generates key points and summaries.

### 5. Cross-Source Analysis

The Analyzer identifies findings, themes, and insights.

### 6. Report Generation

The Writer creates a structured research draft.

### 7. Quality Evaluation

The Critic evaluates:

- Accuracy
- Completeness
- Citation quality

### 8. Revision

If the report does not meet the configured quality threshold, the workflow routes back to the Writer.

### 9. Publishing

Once approved, the Publisher creates the final report.

### 10. Observability

The system returns:

- Final report
- Sources
- Citation metadata
- Critic score
- Revision count
- Events
- Errors

---

# 🏗️ Architecture

## High-Level Architecture

```text
                         ┌───────────────────────┐
                         │       USER QUERY      │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │       ORCHESTRATOR    │
                         │       LangGraph       │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │        PLANNER        │
                         │ 3–5 research subtopics│
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │       RETRIEVER       │
                         │     Tavily Search     │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │         READER        │
                         │ Web extraction + LLM  │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │        ANALYZER       │
                         │ Findings + Themes +   │
                         │       Insights        │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │         WRITER        │
                         │   Research Draft      │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │         CRITIC        │
                         │ Quality Evaluation    │
                         └───────────┬───────────┘
                                     │
                         ┌───────────┴───────────┐
                         │                       │
                    Score ≥ 75              Score < 75
                         │                       │
                         ▼                       ▼
                  ┌─────────────┐        ┌─────────────┐
                  │  PUBLISHER  │        │    WRITER   │
                  │ Final Report│        │   Revision  │
                  └──────┬──────┘        └──────┬──────┘
                         │                       │
                         ▼                       │
                       END ◄─────────────────────┘
```

The actual LangGraph implementation creates the nodes, connects the sequential stages, and adds conditional routing after the Critic. fileciteturn0file0L385-L412

---

# 🔄 End-to-End Workflow

## Step 1 — User Query

The user enters a research question through Streamlit.

Example:

```text
Impact of artificial intelligence on education
```

The Streamlit application provides the query input and a maximum revision-cycle setting. fileciteturn0file1L28-L38

---

## Step 2 — Orchestrator Creates State

The `Orchestrator` creates the initial `ResearchState`.

```python
state = {
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
    "max_revisions": 2,
    "errors": [],
    "events": []
}
```

The Orchestrator then invokes the compiled LangGraph application. fileciteturn0file0L415-L460

---

# 🧠 Agents and Nodes

## 1. 🧠 Planner

### Responsibility

Convert the broad research question into a focused research plan.

The Planner produces:

```python
class plannerOutPut(BaseModel):
    plan_id: str
    subtopics: list[str]
    strategy: str
```

The actual implementation asks the LLM to create 3–5 research subtopics. fileciteturn0file0L99-L115

### Example

Input:

```text
Impact of AI on education
```

Possible plan:

```text
1. AI and personalized learning
2. AI and teacher productivity
3. AI risks in education
4. AI impact on student outcomes
5. Future of AI-powered education
```

---

# 🔎 2. Retriever

### Responsibility

Find relevant online sources for every planned subtopic.

The Retriever calls Tavily for each subtopic and stores:

```python
{
    "title": "...",
    "url": "...",
    "snippet": "..."
}
```

It also records each source in the shared citation store. fileciteturn0file0L118-L152

### Pipeline

```text
Subtopic
   ↓
Tavily Search
   ↓
Search Results
   ↓
URLs + Titles + Snippets
   ↓
Citation Store
```

---

# 📖 3. Reader

The Retriever tells us **where** information is.

The Reader determines **what those sources actually say**.

The Reader uses `trafilatura` to fetch and extract webpage text.

```python
def scrape_url(url):
    downloaded = trafilatura.fetch_url(url)

    if downloaded is None:
        raise ValueError(...)

    text = trafilatura.extract(downloaded)

    if not text:
        raise ValueError(...)

    return text
```

The extracted text is passed to the LLM, which generates:

```python
{
    "url": "...",
    "title": "...",
    "key_points": [],
    "summary": "..."
}
```

The implementation also records failed URLs in `state["errors"]`. fileciteturn0file0L156-L198

---

# 🔬 4. Analyzer

The Analyzer combines the Reader outputs.

It produces:

```python
class AnalayerOutPut(BaseModel):
    findings: list[str]
    themes: list[str]
    insights: list[str]
    citation: list[str]
```

Its job is to identify:

### Findings

Important facts discovered across the sources.

### Themes

Patterns that appear across multiple sources.

### Insights

Higher-level conclusions derived from the collected information.

The Analyzer is explicitly instructed to make findings traceable to the supplied source URLs. fileciteturn0file0L202-L229

---

# ✍️ 5. Writer

The Writer transforms the analysis into a research report.

It receives:

```text
Query
Findings
Themes
Insights
Available Sources
Critic Feedback
```

and produces:

```python
class WriterOutPut(BaseModel):
    draft: str
    citation: list[str]
    reference: list[dict]
```

On a revision pass, the Writer also receives the Critic's feedback and is instructed to address it directly. fileciteturn0file0L233-L275

---

# 🕵️ 6. Critic

The Critic acts as the quality-control layer.

It evaluates the report from 0–100.

```python
class CriticOutPut(BaseModel):
    score: int
    feedback: str
    issues: list[str]
    sugeest: list[str]
    citation_check: Literal[
        "valid",
        "missing",
        "partial"
    ]
```

The Critic evaluates:

```text
Accuracy
Completeness
Citation Quality
```

The actual implementation sets the score range to 0–100. fileciteturn0file0L278-L310

---

# 🔁 Critic → Writer Revision Loop

This is one of the key Agentic AI features.

The Critic does not always terminate the workflow.

Instead:

```text
                 ┌───────────────┐
                 │     Writer    │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │     Critic    │
                 └───────┬───────┘
                         │
                  ┌──────┴──────┐
                  │             │
             Score ≥ 75     Score < 75
                  │             │
                  ▼             ▼
             Publisher       Writer
                                │
                                ▼
                              Critic
```

The routing logic is:

```python
def route_after_critic(state):

    score = state["critic_feedback"]["score"]

    if (
        score >= 75
        or state["revision_count"]
        >= state["max_revisions"]
    ):
        return "publisher"

    state["revision_count"] += 1

    return "writer"
```

This conditional edge is directly implemented in the LangGraph workflow. fileciteturn0file0L401-L409

---

# 📄 7. Publisher

The Publisher is the final stage.

It receives the approved draft and source information.

Its instruction is essentially:

```text
Format the approved research draft.
Do not change facts.
Do not add new claims.
Create a clean final report.
Include references.
```

It then attaches:

```python
metadata = {
    "critic_score": ...,
    "critic_summary": ...,
    "revision_count": ...
}
```

and adds all source metadata from the citation store. fileciteturn0file0L314-L348

---

# 🗃️ Research State

The `ResearchState` is the shared memory of the workflow.

```python
class ResearchState(TypedDict):

    query: str

    search_results: list

    sources: list

    reader_outputs: list

    analysis: dict | None

    report: dict | None

    critic_feedback: dict | None

    final_report: dict | None

    citation_store: dict

    revision_count: int

    max_revisions: int

    errors: list

    events: list

    plan: dict | None
```

This allows every node to access the information generated by previous stages. fileciteturn0file0L63-L77

---

# 🔗 Citation Store

The project includes a shared citation store.

Sources are deduplicated by URL.

Example:

```python
{
    "https://example.com/article": {
        "title": "Example Article",
        "url": "https://example.com/article",
        "source": "web_search",
        "snippet": "Source summary...",
        "fetched_at": "2026-08-23T...",
        "used_in": [
            "retriever",
            "reader"
        ]
    }
}
```

The helper `upsert_citation()` updates existing URLs rather than creating duplicate entries. fileciteturn0file0L28-L46

---

# 🛡️ Retry and Error Handling

The project wraps graph nodes with a retry mechanism.

```text
Node
 │
 ▼
Attempt 1
 │
 ├── Success → Continue
 │
 └── Error
       │
       ▼
    Attempt 2
       │
       ├── Success
       └── Error
             │
             ▼
          Attempt 3
             │
             └── Error → Record failure
```

The `with_retry()` wrapper records:

```text
start
success
retry
error
```

and stores errors in the shared state. fileciteturn0file0L350-L382

---

# 🧭 LangGraph Workflow

The actual graph is:

```python
graph = StateGraph(ResearchState)

graph.add_node("planner", ...)
graph.add_node("retriever", ...)
graph.add_node("reader", ...)
graph.add_node("analyzer", ...)
graph.add_node("writer", ...)
graph.add_node("critic", ...)
graph.add_node("publisher", ...)
```

Sequential edges:

```python
START
    ↓
planner
    ↓
retriever
    ↓
reader
    ↓
analyzer
    ↓
writer
    ↓
critic
```

Conditional routing:

```python
critic
   │
   ├── writer
   │
   └── publisher
```

Final:

```python
publisher → END
```

The compiled graph is exposed as:

```python
app = graph.compile()
```

This is the actual orchestration layer of the project. fileciteturn0file0L385-L412

---

# 🖥️ Streamlit Application

The frontend is built using Streamlit.

The UI provides:

```text
┌──────────────────────────────────────────────┐
│       🔎 Multi-Agent Research Assistant      │
├──────────────────────────────────────────────┤
│                                              │
│ Research Query                               │
│ ┌──────────────────────────────────────────┐ │
│ │ Impact of AI on education                 │ │
│ └──────────────────────────────────────────┘ │
│                                              │
│ Max Revision Cycles: 2                       │
│                                              │
│          [ 🚀 Run Research ]                 │
│                                              │
├──────────────────────────────────────────────┤
│ 📄 Report | 🔗 Sources | 🕵️ Critic | 🛠️ Log │
└──────────────────────────────────────────────┘
```

The actual Streamlit app exposes four tabs:

- 📄 Report
- 🔗 Citations & Sources
- 🕵️ Critic
- 🛠️ Execution Log

and shows API-key status in the sidebar. fileciteturn0file1L28-L38 fileciteturn0file1L79-L128

---

# 🌐 HTML Demonstration

The repository also includes a standalone HTML visualization.

It is useful for demonstrating the architecture without starting Python.

Save the following as:

```text
demo.html
```

```html
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>Multi-Agent AI Research System</title>

<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    font-family: Arial, sans-serif;

    background: #0b1020;

    color: white;

}

header {

    padding: 30px 7%;

    background: #10162a;

    border-bottom: 1px solid #28304a;

}

main {

    width: 92%;

    max-width: 1200px;

    margin: 35px auto;

}

.query {

    display: flex;

    gap: 10px;

    margin-bottom: 30px;

}

input {

    flex: 1;

    padding: 15px;

    border-radius: 10px;

    border: 1px solid #36405e;

    background: #151c32;

    color: white;

}

button {

    padding: 15px 25px;

    border: none;

    border-radius: 10px;

    background: #6d7cff;

    color: white;

    font-weight: bold;

    cursor: pointer;

}

.pipeline {

    display: grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap: 16px;

}

.node {

    background: #151c32;

    border: 1px solid #35405f;

    border-radius: 15px;

    padding: 20px;

    min-height: 150px;

    transition: .3s;

}

.node.active {

    border-color: #8d98ff;

    transform: translateY(-5px);

    box-shadow:
        0 0 25px
        rgba(109,124,255,.3);

}

.status {

    margin-top: 30px;

    padding: 20px;

    border-radius: 15px;

    background: #11182c;

}

.result {

    display: none;

    margin-top: 25px;

    padding: 25px;

    border-radius: 15px;

    background: #151c32;

}

.score {

    font-size: 45px;

    font-weight: bold;

}

@media(max-width:800px) {

    .pipeline {

        grid-template-columns:
            repeat(2, 1fr);

    }

    .query {

        flex-direction: column;

    }

}

</style>

</head>

<body>

<header>

<h1>
🔎 Multi-Agent AI Research System
</h1>

<p>
Planner → Retriever → Reader →
Analyzer → Writer → Critic → Publisher
</p>

</header>

<main>

<div class="query">

<input
    id="query"
    value="Impact of artificial intelligence on education"
>

<button onclick="runDemo()">

Run Research

</button>

</div>


<div class="pipeline">

<div class="node" id="planner">

<h2>🧠 Planner</h2>

<p>
Creates 3–5 research subtopics.
</p>

</div>


<div class="node" id="retriever">

<h2>🔎 Retriever</h2>

<p>
Searches the web with Tavily.
</p>

</div>


<div class="node" id="reader">

<h2>📖 Reader</h2>

<p>
Reads and summarizes webpages.
</p>

</div>


<div class="node" id="analyzer">

<h2>🔬 Analyzer</h2>

<p>
Synthesizes findings across sources.
</p>

</div>


<div class="node" id="writer">

<h2>✍️ Writer</h2>

<p>
Creates the research draft.
</p>

</div>


<div class="node" id="critic">

<h2>🕵️ Critic</h2>

<p>
Evaluates the research report.
</p>

</div>


<div class="node" id="publisher">

<h2>📄 Publisher</h2>

<p>
Creates the final report.
</p>

</div>


<div class="node">

<h2>🔄 Revision</h2>

<p>
Low score → Writer → Critic again.
</p>

</div>

</div>


<div class="status">

<strong>Status:</strong>

<span id="status">
Waiting...
</span>

</div>


<div class="result" id="result">

<h2>🎉 Research Completed</h2>

<p>

Query:

<strong id="resultQuery"></strong>

</p>

<h3>Critic Score</h3>

<div class="score">
82/100
</div>

<p>
Report approved by the Critic.
</p>

</div>

</main>


<script>

const steps = [

"planner",
"retriever",
"reader",
"analyzer",
"writer",
"critic",
"publisher"

];


function sleep(ms) {

return new Promise(

resolve => setTimeout(resolve, ms)

);

}


async function runDemo() {

const query =
document.getElementById("query")
.value.trim();


if(!query) {

alert("Enter a query.");

return;

}


for(const step of steps) {

document
.querySelectorAll(".node")
.forEach(node =>
node.classList.remove("active")
);


document
.getElementById(step)
.classList.add("active");


document
.getElementById("status")
.textContent =
step.toUpperCase()
+ " is running...";


await sleep(900);

}


document
.querySelectorAll(".node")
.forEach(node =>
node.classList.remove("active")
);


document
.getElementById("status")
.textContent =
"Research completed successfully.";


document
.getElementById("resultQuery")
.textContent =
query;


document
.getElementById("result")
.style.display =
"block";

}

</script>

</body>

</html>
```

### HTML Demo Purpose

The HTML is a **visual architecture demonstration**.

It does not execute the real Python agents.

The real research execution happens through:

```text
pipeline.py
    ↓
LangGraph
    ↓
Orchestrator
    ↓
Streamlit app.py
```

---

# 📁 Project Structure

```text
multi-agent-research-system/
│
├── 📄 pipeline.py
│
├── 📄 app.py
│
├── 🌐 demo.html
│
├── 📄 README.md
│
├── 🔐 .env
│
└── 🚫 .gitignore
```

---

## `pipeline.py`

The backend contains:

```text
ResearchState
     ↓
Planner
     ↓
Retriever
     ↓
Reader
     ↓
Analyzer
     ↓
Writer
     ↓
Critic
     ↓
Publisher
     ↓
Orchestrator
```

---

## `app.py`

The Streamlit frontend connects the UI to the compiled graph.

It imports:

```python
from pipeline import Orchestrator, app as compiled_graph
```

and executes:

```python
orchestrator = Orchestrator(
    compiled_graph,
    max_revisions=max_revisions
)

output = orchestrator.run(query)
```

This matches the uploaded Streamlit implementation. fileciteturn0file1L19-L26 fileciteturn0file1L48-L60

---

# ⚙️ Installation

## 1. Clone the Project

```bash
git clone <your-repository-url>

cd multi-agent-research-system
```

## 2. Create Virtual Environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv

source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install python-dotenv
pip install pydantic
pip install beautifulsoup4
pip install langchain-core
pip install langchain-chroma
pip install langchain-tavily
pip install langchain-groq
pip install requests
pip install streamlit
pip install trafilatura
pip install langgraph
```

---

# 🔐 Environment Variables

Create:

```text
.env
```

Add:

```env
GROQ_API_KEY=your_groq_api_key

TAVILY_API_KEY=your_tavily_api_key
```

The pipeline loads these values using `load_dotenv()` and reads the two API keys from the environment. fileciteturn0file0L49-L52

### Never commit `.env`

Use:

```gitignore
.env
.venv/
__pycache__/
```

---

# ▶️ Run the Project

Start Streamlit:

```bash
streamlit run app.py
```

Then open the local Streamlit address shown in the terminal.

---

# 🧪 Example

### Input

```text
Impact of artificial intelligence on education
```

### Internal execution

```text
Query
 ↓
Planner
 ↓
5 Subtopics
 ↓
Tavily Search
 ↓
Multiple Sources
 ↓
Reader
 ↓
Source Summaries
 ↓
Analyzer
 ↓
Findings + Themes + Insights
 ↓
Writer
 ↓
Research Draft
 ↓
Critic
 ↓
Score = 68
 ↓
Revision
 ↓
Writer
 ↓
Critic
 ↓
Score = 84
 ↓
Publisher
 ↓
Final Report
```

---

# 📊 Output

The Orchestrator returns a structured result.

Successful execution:

```python
{
    "ok": True,

    "final_report": {
        "report": "...",
        "reference": [...],
        "citation": [...],
        "metadata": {...},
        "all_sources": [...]
    },

    "revision_count": 1,

    "errors": [],

    "events": [...]
}
```

Failed execution:

```python
{
    "ok": False,

    "error": "...",

    "errors": [...],

    "events": [...]
}
```

The actual Orchestrator implementation follows this success/error structure. fileciteturn0file0L438-L460

---

# 📈 What Makes This an Agentic Workflow?

This project demonstrates several important Agentic AI concepts:

### 🔹 Specialized Roles

Each node has a specific responsibility.

### 🔹 Shared State

Agents communicate through `ResearchState`.

### 🔹 Tool Usage

The Retriever uses web search.

### 🔹 Dynamic Routing

The Critic determines whether the workflow should continue or revise.

### 🔹 Iteration

The Writer can be called multiple times.

### 🔹 Structured Outputs

Pydantic models constrain important LLM outputs.

### 🔹 Error Handling

Failed nodes can retry and record errors.

### 🔹 Observability

The pipeline records execution events.

### 🔹 Source Tracking

URLs and citation metadata are maintained throughout the workflow.

---

# 🚀 Future Improvements

The current project can be extended with:

- ⚡ Parallel source reading
- ⚡ Parallel web search
- 📊 Automated evaluation datasets
- 🧪 LLM-as-a-Judge evaluation
- 🔍 Source credibility scoring
- ✅ Citation verification
- 🧠 Hallucination detection
- 👤 Human-in-the-loop approval
- 📡 Streaming agent events
- 🔭 LangSmith tracing
- 💾 Persistent state
- 🗄️ Vector database integration
- 📄 PDF report export
- 📝 Markdown export
- 🔐 Authentication
- ⚡ Caching
- 📈 Production monitoring

---

# ⚠️ Current Limitations

Based only on the current implementation:

1. The workflow depends on external Groq and Tavily APIs.
2. Webpage extraction can fail for inaccessible or unsupported pages.
3. The Critic is itself an LLM-based evaluator, so its score is not deterministic.
4. The HTML demo is visual only and does not execute the Python backend.
5. The current graph uses sequential node connections except for the Critic routing loop.
6. The configured Groq model may need to be changed if it is unavailable for the user's account.

---

# 🧠 Complete Mental Model

If you want to understand the project in one picture:

```text
                  USER
                   │
                   ▼
              RESEARCH QUERY
                   │
                   ▼
             ┌─────────────┐
             │   PLANNER   │
             └──────┬──────┘
                    │
              Research Plan
                    │
                    ▼
             ┌─────────────┐
             │  RETRIEVER  │
             └──────┬──────┘
                    │
              URLs + Sources
                    │
                    ▼
             ┌─────────────┐
             │    READER   │
             └──────┬──────┘
                    │
            Summaries + Points
                    │
                    ▼
             ┌─────────────┐
             │  ANALYZER   │
             └──────┬──────┘
                    │
          Findings + Themes
                    │
                    ▼
             ┌─────────────┐
             │    WRITER   │
             └──────┬──────┘
                    │
                Draft Report
                    │
                    ▼
             ┌─────────────┐
             │    CRITIC   │
             └──────┬──────┘
                    │
              Quality Score
                 /     \
                /       \
              PASS      FAIL
               │          │
               ▼          ▼
          PUBLISHER     WRITER
               │          │
               ▼          ▼
             FINAL      CRITIC
             REPORT        │
                           └───► ...
```

---

# 👨‍💻 Project Goal

The goal of this project is to demonstrate how a **multi-agent Agentic AI system** can be designed as a real workflow rather than a single LLM call.

The complete system follows:

> **Plan → Search → Read → Analyze → Write → Critique → Revise → Publish**

The backend implementation provides the actual LangGraph orchestration, while the Streamlit application provides the user-facing interface. fileciteturn0file0L385-L412 fileciteturn0file1L15-L17

---

<div align="center">

## ⭐ Multi-Agent AI Research System

### Built with Python + LangGraph + Groq + Tavily + Streamlit

**From a simple research question to an evaluated, source-backed final report.**

</div>
