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

## Taxonomy as config
- **Why a config file, not code or a trained classifier:** Ticket classes change as the business changes. With a YAML file plus an LLM prompt that reads it, a new class works immediately, with no retraining and no code change.
- **Why the loader validates the file:** A typo (duplicate class ID, unknown product, misspelled field) fails at load time with a clear error, instead of causing a strange bug later.
- **Gap I found:** Validation alone wasn't enough, because pydantic ignores unknown keys by default. A misspelled `root_cause` would pass silently. Fixed with `extra="forbid"` and a test.
- **What `held_back` is for:** Two classes (`tv.streaming`, `security.fraud`) are ignored until activated. In the demo I flip one to `active` and show the system handling a new class live.

## Synthetic KB generation
- **LLM writes text, code assigns the rest:** The LLM writes titles, symptoms and steps, while code sets IDs, class labels, product and dates. This keeps the data consistent and gives exact ground truth for evals.
- **Cleaning look-alike characters:** LLMs emit characters like non-breaking hyphens that look normal but aren't. They break keyword search and exact-quote citation checks, so I normalize them to plain ASCII after parsing the JSON.
- **Caveat:** The KB is plausible but not technically validated against real equipment. The assistant relays what the KB says and doesn't judge whether it is correct.

## Questions I should be able to answer
- Why a mock LLM? (Deterministic, free, tests failure modes.)
- Why not cache failures? (A temporary outage would be stored as a permanent answer.)
- Why is the data decision a strength? (I checked the data before building on it, and I document what is real and what is synthetic.)

- Why do customer typos matter? (Embeddings handle them fairly well, keyword search doesn't. I'll measure clean vs noisy Recall@5 instead of adding a blind spell-corrector, which could damage error codes.)

## test need 
A function that silently returns None is a classic bug. The tests caught it within seconds, which is exactly what they are for. If you had skipped running them, the generator would have crashed halfway through 40 API calls.


## Checking assumptions against data
- **What happened:** After reading two synthetic tickets, I assumed the `steps_already_tried` field was often wrong, because both complaints mentioned a tried step that the field didn't list.
- **What I did:** I wrote a small script (`scripts/check_tried_steps.py`) to measure it instead of trusting two samples.
- **Result:** 141 English complaints mention a tried step, and only 23 of them (16%) have an empty field. The pattern was much smaller than I thought.
- **Decision:** Keep using the field as a partial ground truth, state the 16% gap in `DATA.md`, and have the parser extract tried steps from the complaint text. Parser tests use hand-written complaints.
- **Lesson:** Measure before documenting a pattern. A rough check (keyword matching) is still better than a guess, as long as I say it is rough.

## Questions I should be able to answer
- Why not rely on the `steps_already_tried` field at runtime? (A real system only sees the complaint text, so the parser has to read it from there.)
- How reliable is the synthetic data? (Mostly consistent, with measured gaps listed in `DATA.md`.)