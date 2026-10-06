# Nexolve Resolution Copilot

<p align="center">
  <img src="docs/assets/nexolve-banner.svg" alt="Nexolve Resolution Copilot - from complaint to confident next step" width="100%" />
</p>

<p align="center">
  <img alt="API" src="https://img.shields.io/badge/API-FastAPI-079A82?style=flat-square" />
  <img alt="Frontend" src="https://img.shields.io/badge/UI-React%20%2B%20Vite-5B67E8?style=flat-square" />
  <img alt="Retrieval" src="https://img.shields.io/badge/Retrieval-Semantic%20%2B%20BM25-6941C6?style=flat-square" />
  <img alt="Answers" src="https://img.shields.io/badge/Answers-Cited%20%26%20Grounded-CC8A25?style=flat-square" />
</p>


<p align="center">
  <strong>Turn a customer complaint into evidence-backed support guidance.</strong><br/>
  A semantic support assistant for telecom service teams.
</p>

<p align="center">
  <em>Understand the issue · Find relevant history · Draft a grounded next step</em>
</p>

---

## Why this project exists

Support agents often search old cases with a few keywords. That works when customers use the same words as the ticket title; it misses when they describe the same fault differently.

A customer might say, “My broadband drops every evening while I’m working.” A useful assistant should recognize the connectivity problem, retrieve relevant support articles and resolved cases, and help the agent respond with clear steps grounded in those sources.

Nexolve Resolution Copilot is designed to make that workflow faster while keeping the evidence visible. It is an assistant for support staff: retrieved sources inform the recommendation, and the agent remains responsible for reviewing it.

## What it does

- Accepts a free-text customer complaint in a web interface.
- Extracts useful complaint signals such as product, issue category, severity, and sentiment.
- Searches support knowledge and historical resolved tickets using semantic and lexical retrieval.
- Uses query-aware feedback signals to adjust rankings for similar future complaints.
- Drafts a step-by-step response from retrieved evidence.
- Returns visible source references and validates citations against retrieved evidence.
- Applies telecom scope and response guardrails, with safe handling when evidence is insufficient.
- Captures helpfulness feedback and exposes service health and request diagnostics.

## Architecture

The diagram shows the intended request and feedback flow. It intentionally omits dataset counts.

```mermaid
flowchart TB
    CUSTOMER["Customer complaint"] --> UI["React + Vite interface"]
    UI -->|"POST /resolve"| API["FastAPI service"]
    API --> SCOPE{"Telecom scope check"}
    SCOPE -->|Out of scope| SAFE["Safe response; skip retrieval"]
    SCOPE -->|In scope| QUERY["Prepare complaint query"]

    subgraph RETRIEVAL["Evidence retrieval"]
        direction TB
        QUERY --> SEARCH["Hybrid search"]
        SEARCH --> SEM["Semantic similarity"]
        SEARCH --> BM25["BM25 keyword ranking"]
        SEM --> RANK["Merge and rank candidates"]
        BM25 --> RANK
        RANK --> FEEDBACK_RANK["Query-aware feedback reranking"]
        FEEDBACK_RANK --> CONTEXT["Build evidence context"]
    end

    KB[("Knowledge base")] --> SEARCH
    TICKETS[("Resolved ticket history")] --> FILTER["Filter unsuitable cases<br/>Redact personal information"]
    FILTER --> SEARCH

    CONTEXT --> LLM["LLM drafts from retrieved evidence"]
    LLM --> VALIDATE["Citation validation<br/>Output guardrails"]
    VALIDATE --> RESULT["Resolution with cited sources"]
    SAFE --> RESULT
    RESULT --> UI

    UI --> RATING["Helpful / not helpful"]
    RATING --> FEEDBACK_API["POST /feedback"]
    FEEDBACK_API --> STORE[("Feedback store")]
    STORE -.-> FEEDBACK_RANK

    API -.-> OPS["Health checks<br/>Request and trace logs"]

    classDef user fill:#eef4ff,stroke:#5278d0,color:#172554,stroke-width:1.5px;
    classDef service fill:#f4f0ff,stroke:#8064c8,color:#24134f,stroke-width:1.5px;
    classDef retrieval fill:#e9f8f4,stroke:#36977c,color:#123b32,stroke-width:1.5px;
    classDef data fill:#fff6e8,stroke:#d49b42,color:#50380d,stroke-width:1.5px;
    classDef safety fill:#fff0f0,stroke:#d66b6b,color:#521f1f,stroke-width:1.5px;

    class CUSTOMER,UI,RESULT,RATING user;
    class API,LLM,CONTEXT,OPS service;
    class QUERY,SEARCH,SEM,BM25,RANK,FEEDBACK_RANK retrieval;
    class KB,TICKETS,FILTER,STORE data;
    class SCOPE,SAFE,VALIDATE safety;
```

### Request lifecycle

1. The customer complaint enters through the web application.
2. The API checks whether the issue is in the supported telecom domain.
3. For an in-scope complaint, the retrieval layer searches support articles and eligible resolved cases using semantic and BM25 signals.
4. Feedback from similar queries can influence ranking. Retrieved evidence is assembled as the only factual basis for the generated recommendation.
5. The answer is checked for valid source citations and safe output before it is returned.
6. The UI presents the resolution and its knowledge sources; the customer or agent can rate whether it helped.

## Grounding and safety

The assistant should prefer a useful abstention over a confident answer without evidence.

- **Source-grounded generation:** recommendations should be based on retrieved material, not invented telecom procedures.
- **Citation validation:** source identifiers in the answer must correspond to sources actually retrieved for that request.
- **Evidence threshold:** if retrieval does not find sufficiently relevant material, the response should explain the evidence gap and avoid specific unsupported troubleshooting.
- **Domain scope:** non-telecom complaints should be routed to a safe out-of-scope response without presenting unrelated telecom articles as relevant.
- **Ticket hygiene:** exclude unsuitable historical cases and remove personal information before indexing or using ticket text.
- **Human review:** agents should verify actions that affect accounts, billing, equipment, or service status.

## Additional exploration

The system is designed to evolve as customer language, products, and ticket classes change.

### Retrieval experiments

Compare lexical BM25, semantic similarity, and hybrid retrieval on the same labeled complaint set. Include paraphrases, spelling mistakes, short complaints, and complaints with multiple symptoms. Track which source type—article or historical case—provides the strongest evidence.

### Feedback-aware ranking

Use helpful and not-helpful feedback as a ranking signal for semantically similar future queries. Keep this signal bounded: feedback should adjust candidate ordering, not override relevance, source quality, domain scope, or safety rules. Log the ranking version so experiments can be compared and rolled back.

### Evolving taxonomy and data

Monitor new product names, issue categories, and emerging ticket clusters. Add a controlled ingestion path that validates records, redacts sensitive fields, deduplicates cases, and refreshes indexes. Version the corpus and taxonomy so results can be traced to the data that produced them.

### Out-of-domain and low-evidence behavior

Include deliberate negative examples—such as vehicle repair, medical, or unrelated consumer questions—in evaluation. The expected behavior is to decline domain-specific guidance and avoid displaying unrelated retrieved sources as supporting evidence.

## Evaluation: quality and system health

The figures below are taken from the evaluation artifacts in your project folder (data/eval/retrieval_results.json and data/eval/answer_results.jsonl). The retrieval results are a measured run of the project hybrid retriever on its synthetic labeled benchmark; they are not production traffic metrics.

## 📊 Evaluation Results

The system was evaluated across retrieval quality, retrieval latency, and
citation validity.

| Metric | Observed Result | What it measures |
|---|---:|---|
| **Recall@1 (R@1)** | **56.75%** | Percentage of queries where the relevant article was ranked first |
| **Recall@3 (R@3)** | **77.00%** | Percentage of queries where the relevant article appeared in the top 3 |
| **Recall@5 (R@5)** | **84.25%** | Percentage of queries where the relevant article appeared in the top 5 |
| **nDCG@5** | **0.7146** | Quality of the ranking within the top 5 results |
| **Retrieval Latency** | **p50: 46.2 ms · p95: 91.1 ms** | Retrieval-stage latency in the recorded offline run |
| **Citation Validity** | **70% (14/20)** | Percentage of evaluated answers that contained valid citations to retrieved evidence |

### Evaluation Scope

- **Retrieval:** 400 synthetic complaint queries
- **Latency:** recorded offline retrieval run
- **Citation evaluation:** 20 generated answers (testing purpose)
- Citation validity checks whether cited knowledge-base IDs correspond to
  retrieved evidence.

> **Note:** Retrieval metrics and citation validity measure different stages of
> the pipeline and should not be interpreted as a single overall accuracy
> score.

## Production scale considerations

### ## 🏗️ Production-Grade Resolution Design

A reliable support assistant needs more than a *retrieve → prompt → LLM* pipeline. Nexolve uses multiple stages to find relevant evidence, protect customer information, account for agent feedback, validate generated guidance, and handle cases where the evidence is not strong enough.

### ⚡ Weighted hybrid retrieval
Nexolve combines dense semantic search with BM25 lexical retrieval. Semantic search can recognize complaints that describe the same fault in different words, while lexical search helps surface exact product names, error messages, and technical terms. Weighted results bring both kinds of matches into the ranking.

### 🔄 Query-aware feedback reranking
Helpfulness feedback is considered in the context of the current complaint. Feedback from similar past queries can strengthen relevant candidates, while feedback from unrelated issues has less influence. This helps improve ranking without turning globally popular articles into default answers.

### 🎫 Past-ticket retrieval
Resolved support cases provide a second source of evidence alongside knowledge-base articles. Similar tickets can reveal how related issues were investigated and resolved, while the knowledge base provides reusable support guidance.

### 🔒 PII redaction
Personal information in ticket data is redacted before it is indexed or used as retrieval context. This reduces the risk of exposing customer details in search results, prompts, logs, or generated responses.

### 🔗 Citation validation
Retrieved source IDs are carried through the generation process. Before an answer is returned, its citations are checked against the sources retrieved for that request. This helps prevent the assistant from citing articles or tickets it did not actually use.

### 🛡️ RAG guardrails
The system checks whether the retrieved evidence is relevant and sufficient, then validates the generated response. When an output fails validation, it can attempt a controlled repair. If the evidence still cannot support a safe, specific recommendation, the assistant returns a limited-evidence response instead.

### ⚙️ Failure-aware LLM pipeline
LLM providers can time out, apply rate limits, or become temporarily unavailable. The pipeline handles these failures explicitly and avoids silently presenting unsupported guidance as a successful resolution.

> **Design principle**  
> *Redact sensitive data → retrieve broadly → rerank for the current query → generate from retrieved evidence → validate the answer and citations → abstain when evidence is insufficient.*


## Run locally

Clone the repository, then choose either Docker or the manual setup.

```powershell
git clone <your-repository-url>
cd nexolve-resolution-copilot
```

Create a `.env` file from the project’s example, if provided, and add the required LLM credentials. Never commit secrets.

### Option 1: Docker

Make sure Docker Desktop is running. From the project root, run:

```powershell
docker compose up --build
```

Open **http://localhost:5173**. Stop the services with `Ctrl+C`, or run `docker compose down` from the project directory.

This option requires a `compose.yaml` or `docker-compose.yml` in the repository, configured to expose the frontend on port `5174` and connect it to the API.

### Option 2: Manual setup

Create and activate a Python virtual environment, then install the backend dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

In one terminal, start the API from the project root:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal, start the frontend:

```powershell
cd frontend
npm install
npm run dev -- --port 5173
```

Open **http://localhost:5173**. Configure the frontend to reach the API at **http://127.0.0.1:8000**. Follow the project files for the required Python version and environment variables.
shows both source metadata and citations, and helpfulness feedback is accepted.

---

<p align="center">
  <sub>Evidence first. Clear next steps. Better support conversations.</sub>
</p>


