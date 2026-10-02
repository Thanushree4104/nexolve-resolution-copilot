## Chunking: one vector per article vs per symptom
Same data, one run. Per-symptom vectors take the best-matching symptom per article.

| Variant | R@1 | R@5 | MRR | class@1 |
|---|---|---|---|---|
| One vector per article | 0.37 | 0.74 | 0.54 | 0.65 |
| Per symptom, with title | 0.35 | 0.70 | 0.51 | 0.65 |
| Per symptom, no title | 0.14 | 0.52 | 0.32 | 0.36 |

**Findings:** Per-symptom chunking did not help. Removing the title hurt a lot, so the title carries most of the class signal. Caveat: symptoms were generated to avoid the title's wording, which makes them weak evidence alone.
**Decision:** One vector per article.

**Error analysis:** In 6 manually read errors, a human would have chosen the correct problem class every time (for example "no internet at all" was matched to an intermittent-connection article), so these are retrieval weaknesses and not label overlap. Hypotheses: long embedded text dilutes the title, and a few articles sit near every complaint. Within a class, the root cause is often genuinely ambiguous from the complaint alone.

## Oracle class: how much would a perfect classifier add?
Candidates restricted to the true class (4 articles). Random: R@1 0.25, R@2 0.50, MRR 0.52.

| Method | R@1 | R@2 | MRR |
|---|---|---|---|
| Dense | 0.54 | 0.77 | 0.72 |
| BM25 | 0.47 | 0.73 | 0.68 |
| Hybrid | 0.52 | 0.77 | 0.71 |

**Findings:** A perfect class prediction would lift R@1 from 0.40 (best unrestricted) to 0.54, an upper bound. Within a class, hybrid does not beat dense, so its English gain probably came from the class decision. Even with the class known, the right article is first only about half the time but in the top two 77% of the time, which motivates clarifying questions. Single run, synthetic data, 53 German tickets.