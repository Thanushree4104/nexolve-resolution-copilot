## Chunking: one vector per article vs per symptom
Same data, one run. Per-symptom vectors take the best-matching symptom per article.

| Variant | R@1 | R@5 | MRR | class@1 |
|---|---|---|---|---|
| One vector per article | 0.37 | 0.74 | 0.54 | 0.65 |
| Per symptom, with title | 0.35 | 0.70 | 0.51 | 0.65 |
| Per symptom, no title | 0.14 | 0.52 | 0.32 | 0.36 |

**Findings:** Per-symptom chunking did not help. Removing the title hurt a lot, so the title carries most of the class signal. Caveat: symptoms were generated to avoid the title's wording, which makes them weak evidence alone.
**Decision:** One vector per article.