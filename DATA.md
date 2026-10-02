# Data

## Real data
- **Source:** Hugging Face `Tobi-Bueck/customer-support-tickets` (61,765 rows, English and German, CC-BY-NC-4.0).
- **Not committed.** Downloaded by `scripts/download_data.py` into `data/raw/`.
- **Finding:** About 69% of agent replies in the network-related subset are information requests, and none of 12 sampled replies contained concrete troubleshooting steps. Most "network" tickets are also not telecom (SaaS, marketing, finance).
- **Use:** Profiling and the data-quality finding. Planned: realistic complaint wording, German phrasing, and mining the diagnostic questions agents ask.

## Synthetic data
All generated with `openai/gpt-oss-120b` via Groq on 2026-10-01 to 02, with fixed seeds for everything assigned by code.

| File | Contents |
|---|---|
| `data/synthetic/kb_articles.jsonl` | 32 KB articles for the 8 active ticket classes |
| `data/synthetic/kb_heldback.jsonl` | 8 KB articles for 2 held-back classes (used in the live "new class" demo) |
| `data/synthetic/tickets.jsonl` | 400 resolved tickets (50 per active class) |

### Split of responsibilities
- **The LLM writes text:** KB articles, complaints, resolution notes.
- **Code assigns structure:** IDs, class, product, root cause, language, wording style, persona, resolution quality, steps already tried, reopened flag, first-contact flag, dates, region, firmware, and the KB article each ticket should map to (`kb_id`, used as eval ground truth).

### Distributions (tickets)
- Language: 347 English, 53 German
- Resolution quality: 209 detailed, 128 brief, 63 vague
- Reopened: 39 (about 10%), more likely when the resolution note is vague

## Limitations
- Operator, products and firmware are fictional.
- KB values (thresholds, power levels, procedures) are plausible but not validated against real equipment.
- Generated text is cleaner and more uniform than real tickets.
- Only 53 German tickets, so German metrics are noisy.
- Not yet included: Hindi and Tamil text, an incident cluster, typo-noised evaluation queries.


- Product and firmware were assigned independently of root cause, so some tickets pair a device-specific cause (for example "firmware bug in the ONT") with the generic Nimbus Fibre product and no firmware version.
- The LLM sometimes adds steps the customer "already tried" that are not in the `steps_already_tried` field, so that field is a lower bound.


- In about 16% of English complaints that mention a tried step (23 of 141, by a rough keyword check), the `steps_already_tried` field is empty, because the LLM added steps beyond its seed. The field is therefore a lower bound and is not used as the only ground truth for the parser. Parser tests use hand-written complaints.