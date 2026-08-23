# Multi-AI-Research_System

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Multi-Agent AI Research System</title>

<style>
    * {
        box-sizing: border-box;
    }

    body {
        margin: 0;
        font-family: Arial, sans-serif;
        background: #0b1020;
        color: #e8ecf7;
    }

    header {
        padding: 30px 7%;
        background: #10162a;
        border-bottom: 1px solid #28304a;
    }

    header h1 {
        margin: 0 0 8px;
    }

    .subtitle {
        color: #aab4cf;
    }

    main {
        width: min(1200px, 92%);
        margin: 35px auto;
    }

    .query-box {
        display: flex;
        gap: 12px;
        margin-bottom: 35px;
    }

    input {
        flex: 1;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #36405e;
        background: #151c32;
        color: white;
        font-size: 16px;
    }

    button {
        padding: 15px 25px;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        background: #6d7cff;
        color: white;
        font-weight: bold;
    }

    button:hover {
        opacity: 0.85;
    }

    .pipeline {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
    }

    .node {
        background: #151c32;
        border: 1px solid #35405f;
        border-radius: 14px;
        padding: 20px;
        min-height: 150px;
        transition: 0.3s;
    }

    .node.active {
        border-color: #8d98ff;
        transform: translateY(-5px);
        box-shadow: 0 0 25px rgba(109,124,255,0.3);
    }

    .number {
        width: 35px;
        height: 35px;
        display: grid;
        place-items: center;
        border-radius: 50%;
        background: #252e4d;
        margin-bottom: 12px;
        font-weight: bold;
    }

    .node h3 {
        margin: 5px 0;
    }

    .node p {
        color: #aab4cf;
        font-size: 14px;
        line-height: 1.5;
    }

    .flow {
        margin: 35px 0;
        text-align: center;
        color: #aab4cf;
        line-height: 2;
    }

    .status {
        margin-top: 30px;
        background: #11182c;
        border: 1px solid #2c3653;
        border-radius: 14px;
        padding: 20px;
    }

    .status strong {
        color: #aeb7ff;
    }

    .result {
        display: none;
        margin-top: 25px;
        background: #151c32;
        border: 1px solid #35405f;
        border-radius: 14px;
        padding: 25px;
    }

    .score {
        font-size: 45px;
        font-weight: bold;
    }

    .revision {
        border: 1px dashed #596580;
    }

    footer {
        text-align: center;
        color: #7884a8;
        padding: 35px;
    }

    @media (max-width: 800px) {
        .pipeline {
            grid-template-columns: repeat(2, 1fr);
        }

        .query-box {
            flex-direction: column;
        }
    }
</style>
</head>

<body>

<header>

    <h1>🔎 Multi-Agent AI Research System</h1>

    <div class="subtitle">
        Planner → Retriever → Reader → Analyzer → Writer → Critic → Publisher
    </div>

</header>


<main>

    <!-- QUERY -->

    <div class="query-box">

        <input
            id="query"
            value="Impact of artificial intelligence on education"
            placeholder="Enter research query..."
        >

        <button onclick="runDemo()">
            Run Research
        </button>

    </div>


    <!-- PIPELINE -->

    <section class="pipeline">


        <!-- PLANNER -->

        <div class="node" id="planner">

            <div class="number">1</div>

            <h3>🧠 Planner</h3>

            <p>
                Breaks the research question into
                3–5 focused research subtopics.
            </p>

        </div>


        <!-- RETRIEVER -->

        <div class="node" id="retriever">

            <div class="number">2</div>

            <h3>🔎 Retriever</h3>

            <p>
                Searches the web using Tavily
                and collects relevant sources.
            </p>

        </div>


        <!-- READER -->

        <div class="node" id="reader">

            <div class="number">3</div>

            <h3>📖 Reader</h3>

            <p>
                Fetches webpages and extracts
                useful information and summaries.
            </p>

        </div>


        <!-- ANALYZER -->

        <div class="node" id="analyzer">

            <div class="number">4</div>

            <h3>🔬 Analyzer</h3>

            <p>
                Combines information from multiple
                sources into findings and insights.
            </p>

        </div>


        <!-- WRITER -->

        <div class="node" id="writer">

            <div class="number">5</div>

            <h3>✍️ Writer</h3>

            <p>
                Generates a structured research
                report using the analyzed findings.
            </p>

        </div>


        <!-- CRITIC -->

        <div class="node" id="critic">

            <div class="number">6</div>

            <h3>🕵️ Critic</h3>

            <p>
                Evaluates accuracy, completeness
                and citation quality.
            </p>

        </div>


        <!-- PUBLISHER -->

        <div class="node" id="publisher">

            <div class="number">7</div>

            <h3>📄 Publisher</h3>

            <p>
                Formats the approved draft into
                the final research report.
            </p>

        </div>


        <!-- REVISION -->

        <div class="node revision" id="revision">

            <div class="number">↻</div>

            <h3>🔄 Revision Loop</h3>

            <p>
                If the critic score is below 75,
                feedback goes back to the Writer.
            </p>

        </div>


    </section>


    <!-- FLOW -->

    <div class="flow">

        START
        →
        Planner
        →
        Retriever
        →
        Reader
        →
        Analyzer
        →
        Writer
        →
        Critic

        <br>

        Critic →

        <strong>Score ≥ 75 → Publisher</strong>

        <br>

        Critic →

        <strong>Score &lt; 75 → Writer → Critic</strong>

        <br>

        Publisher → END

    </div>


    <!-- STATUS -->

    <section class="status">

        <strong>Status:</strong>

        <span id="status">
            Waiting for research query...
        </span>

    </section>


    <!-- RESULT -->

    <section class="result" id="result">

        <h2>📄 Research Completed</h2>

        <p>

            <strong>Research Query:</strong>

            <span id="resultQuery"></span>

        </p>

        <p>

            <strong>Pipeline:</strong>

            Planner → Retriever → Reader →
            Analyzer → Writer → Critic → Publisher

        </p>

        <h3>Critic Evaluation</h3>

        <div class="score">
            <span id="score">82</span>/100
        </div>

        <p>
            The report passed the critic threshold.
            The Publisher can now produce the final report.
        </p>

    </section>

</main>


<footer>

    Multi-Agent AI Research System

    <br>

    LangGraph · Groq · Tavily · Streamlit

</footer>


<script>

const steps = [

    "planner",
    "retriever",
    "reader",
    "analyzer",
    "writer",
    "critic"

];


function sleep(ms) {

    return new Promise(
        resolve => setTimeout(resolve, ms)
    );

}


async function runDemo() {

    const query =
        document.getElementById("query")
        .value
        .trim();


    if (!query) {

        alert("Enter a research query.");

        return;

    }


    // Hide previous result

    document.getElementById("result")
        .style.display = "none";


    // Run every agent

    for (const id of steps) {


        // Remove previous active state

        document
            .querySelectorAll(".node")
            .forEach(node => {

                node.classList.remove("active");

            });


        // Activate current node

        document
            .getElementById(id)
            .classList
            .add("active");


        const messages = {

            planner:
                "🧠 Planner is creating the research plan...",

            retriever:
                "🔎 Retriever is searching the web...",

            reader:
                "📖 Reader is reading source webpages...",

            analyzer:
                "🔬 Analyzer is synthesizing information...",

            writer:
                "✍️ Writer is generating the research draft...",

            critic:
                "🕵️ Critic is evaluating the research draft..."

        };


        document
            .getElementById("status")
            .textContent = messages[id];


        await sleep(1000);

    }


    // Remove active states

    document
        .querySelectorAll(".node")
        .forEach(node => {

            node.classList.remove("active");

        });


    // Publisher

    document
        .getElementById("publisher")
        .classList
        .add("active");


    document
        .getElementById("status")
        .textContent =
        "✅ Critic approved the report → Publisher is formatting it...";


    await sleep(1200);


    // Complete

    document
        .getElementById("publisher")
        .classList
        .remove("active");


    document
        .getElementById("status")
        .textContent =
        "🎉 Research pipeline completed successfully.";


    document
        .getElementById("resultQuery")
        .textContent = query;


    document
        .getElementById("result")
        .style
        .display = "block";

}

</script>

</body>
</html>
