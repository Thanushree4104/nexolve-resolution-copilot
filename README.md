<div align="center">

# 🧠 Nexolve Resolution Copilot

**Grounded, citation-backed resolutions for customer complaints, powered by Hybrid RAG.**

![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791?logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![LLM](https://img.shields.io/badge/LLM-Groq-F55036)
![License](https://img.shields.io/badge/License-MIT-green)

[Quick Start](#-quick-start) · [API Usage](#-api-usage) · [Architecture](#-architecture) · [How It Works](#-how-it-works)

</div>

---

## 📌 Overview

Nexolve Resolution Copilot helps support teams respond to customer complaints faster and more consistently. It analyzes an incoming complaint, retrieves the most relevant **knowledge-base articles** and **similar past tickets**, and generates a resolution that is **grounded in those sources and cited**, not guessed.

**Why it matters:** support agents get a trustworthy first draft, every claim traceable to a source, with guardrails to stop unsupported answers from reaching the customer.

## ✨ Key Features

- **Complaint analysis:** extracts intent, product, severity, and sentiment.
- **Hybrid retrieval:** combines semantic (vector) search with BM25 keyword search.
- **Two knowledge sources:** curated knowledge base plus historical ticket resolutions.
- **Grounded generation:** the LLM answers only from retrieved context.
- **Citations and guardrails:** responses are validated against their sources.
- **Evolving data:** new tickets and KB content improve retrieval over time.
- **One-command setup:** fully containerized with Docker Compose.

## 🏗️ Architecture

```text
Customer Complaint
       ↓
    FastAPI
       ↓
Complaint Analysis
(Intent • Product • Severity • Sentiment)
       ↓
   Hybrid RAG
   ┌────┴─────┐
   ↓          ↓
Knowledge   Past Tickets
Base        Retrieval
   └────┬─────┘
        ↓
PostgreSQL + pgvector
        ↓
   Context Builder
        ↓
       LLM
        ↓
Citation + Guardrails
        ↓
Grounded Resolution
```

> **PostgreSQL + pgvector** provides persistent storage for knowledge-base and historical ticket embeddings, enabling semantic retrieval and supporting the system as new tickets and knowledge-base content are added.

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI |
| LLM | Groq |
| Embeddings | Sentence Transformers |
| Vector DB | PostgreSQL + pgvector |
| Retrieval | Hybrid Semantic + BM25 |
| RAG | Custom RAG pipeline |
| Containerization | Docker Compose |

## 📁 Project Structure

```text
.
├── app/
│   ├── api/            # FastAPI routes
│   ├── analysis/       # Complaint analysis (intent, product, severity, sentiment)
│   ├── retrieval/      # Hybrid semantic + BM25 retrieval
│   ├── generation/     # Context builder, LLM calls, citations, guardrails
│   └── db/             # PostgreSQL + pgvector access
├── data/               # Knowledge base and sample tickets
├── docs/               # Deep-dive documentation
├── docker-compose.yml
├── Dockerfile
├── .env.example
└── README.md
```

> Adjust folder names to match your repository layout.

## 🚀 Quick Start

### Prerequisites

- Docker
- Docker Compose
- Groq API key

### 1. Configure environment

Copy `.env.example` to `.env` and add your Groq API key.

```bash
cp .env.example .env
```

### 2. Start the application

```bash
docker compose up --build
```

The application will be available at:

**http://localhost:8000**

Interactive API docs (Swagger UI): **http://localhost:8000/docs**

## 🔌 API Usage

**Request**

```bash
curl -X POST http://localhost:8000/resolve \
  -H "Content-Type: application/json" \
  -d "{\"complaint\":\"My broadband keeps dropping every evening around 8 PM.\"}"
```

**Response (illustrative)**

```json
{
  "analysis": {
    "intent": "connectivity_issue",
    "product": "broadband",
    "severity": "medium",
    "sentiment": "frustrated"
  },
  "resolution": "Intermittent evening drops are commonly caused by network congestion... [KB-12] A similar case was resolved by... [TICKET-348]",
  "citations": ["KB-12", "TICKET-348"]
}
```

> Field names are illustrative. Check `/docs` for the exact schema.

## ⚙️ How It Works

1. **Analyze:** the complaint is classified by intent, product, severity, and sentiment.
2. **Retrieve:** hybrid search (semantic + BM25) pulls from both the knowledge base and past tickets.
3. **Build context:** top results are merged into a compact, source-labelled context.
4. **Generate:** the LLM drafts a resolution using only that context.
5. **Validate:** citations are checked and guardrails reject or repair unsupported output.

## 🔄 Evolving Data

The system is designed to grow with your support operation:

- New **knowledge-base articles** are embedded and indexed for retrieval.
- Newly resolved **tickets** become searchable precedent.
- Because storage lives in PostgreSQL + pgvector, data **persists across restarts**.

## 📊 Evaluation / Results

| Metric | Result |
|---|---|
| Retrieval relevance (Top-k) | _TBD_ |
| Citation validity rate | _TBD_ |
| Grounded-answer rate | _TBD_ |
| Avg. response latency | _TBD_ |

> Fill in with your measured numbers before sharing with reviewers.

## 📚 Further Reading

Deeper details are kept out of this README on purpose. See the [`docs/`](docs/) folder:

- Citation validation rules
- Guardrail repair logic
- Feedback reranking internals
- Database schema
- Retrieval algorithms

## 📄 License

Released under the [MIT License](LICENSE).
