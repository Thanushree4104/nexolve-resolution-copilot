# Learning Log

## Project description
## Project in one line
A support assistant for telecom agents: the agent pastes a customer complaint into a simple screen and gets a step-by-step fix with cited sources, built from similar past tickets and KB articles. It runs on a microservice backend that handles parsing, retrieval, drafting and verification.

## Assistant vs microservice
- **Assistant:** the agent-facing screen (planned in Streamlit) where the complaint is pasted and the answer, sources and warnings are shown.
- **Microservice:** the FastAPI backend (`/v1/assist` and others) that does the work.
- **Why split them:** the backend can be tested, scaled, monitored and reused on its own, and the screen stays a thin layer on top.
- **Decision:** the brief asked for an assistant, so the UI is part of the core deliverable and not a late extra. I build it right after the first working `/v1/assist` endpoint.

## Why both an assistant and an API? 
- The agent needs a usable screen, and the service behind it needs to be testable, observable and scalable independently.

## Request ID and structured logging
- **What it is:** Every request gets a unique ID, returned in the `X-Request-ID` header and included in every JSON log line.
- **Why:** Lets me follow one request through the whole system (traceability), and JSON logs can be searched and aggregated (observability).
- **Detail:** If a caller sends its own ID, I reuse it (capped at 64 characters, since it comes from outside). A `ContextVar` carries the ID so I don't pass it through every function.

## Settings in one place
- **What it is:** One `Settings` class reads values from environment variables and `.env`.
- **Why:** No scattered `os.environ` calls, easy to override per environment, and secrets stay out of the code and git.

## LLM provider interface
- **What it is:** A common `complete(messages)` method that every LLM backend implements (mock, Groq, and possibly Ollama later).
- **Why:**
  - Swap backends with a config change.
  - Tests and CI run without an API key, using the mock.
  - I can simulate an LLM outage to test fallbacks.
  - Caching, retries and metrics go in one layer.

## LLMUnavailableError vs LLMError
- **Difference:** `LLMUnavailableError` means the LLM is down, rate-limited or timing out (the request was fine). `LLMError` means the request or output is bad (wrong model name, empty answer).
- **What the system does:**
  - Unavailable: fall back (return ranked sources without a drafted answer, marked `degraded`).
  - Error: don't pretend it's an outage, since retrying won't help. Log it and fix the cause.
- **Why separate:** Otherwise a bug looks like an outage and gets hidden, or a real outage crashes the request.

## Disk cache for LLM calls
- **What it is:** A wrapper that stores each response on disk, keyed by a hash of the model, messages and settings.
- **Why:** Saves free-tier quota, makes reruns fast and reproducible, and failures are never cached.
- **Bug I found:** Cache hits replayed the original 900 ms latency, which would distort metrics. Fixed by measuring the real cache-read time. Lesson: test what the numbers mean, not just that code runs.

## Data findings
- Profiled 61K real tickets (54% German, 46% English). About 12.8K are network-related, and about 69% of those answers are information requests.
- Read 12 sample answers: none contained concrete troubleshooting steps, and most "network" tickets were not telecom.
- **Decision:** Don't build resolutions from this data. Use the real tickets for complaint wording, German text, and mining agents' diagnostic questions, and generate telecom resolutions and KB articles synthetically.

## Questions I should be able to answer
- Why a mock LLM? (Deterministic, free, tests failure modes.)
- Why not cache failures? (A temporary outage would be stored as a permanent answer.)
- Why is the data decision a strength? (I checked the data before building on it, and I document what is real and what is synthetic.)
