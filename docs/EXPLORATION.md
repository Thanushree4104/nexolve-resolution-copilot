# Retrieval Exploration

## Chunking: One Vector per Article vs. Per Symptom

We compared two retrieval representations using the same dataset and a single
run.

For the per-symptom representation, each article is represented by multiple
symptom vectors, and the best-matching symptom for each article is used for
evaluation.

| Variant | R@1 | R@5 | MRR | class@1 |
|---|---:|---:|---:|---:|
| One vector per article | **0.37** | **0.74** | **0.54** | **0.65** |
| Per symptom, with title | 0.35 | 0.70 | 0.51 | **0.65** |
| Per symptom, no title | 0.14 | 0.52 | 0.32 | 0.36 |

### Findings

Per-symptom chunking did **not** improve retrieval performance.

The strongest degradation occurred when the article title was removed:

- R@1 dropped from **0.35 → 0.14**
- R@5 dropped from **0.70 → 0.52**
- MRR dropped from **0.51 → 0.32**
- class@1 dropped from **0.65 → 0.36**

This indicates that the **title carries substantial class-level signal** in this
dataset.

There is an important caveat: the symptoms were generated specifically to
avoid repeating the wording of the article title. As a result, symptom text
alone provides relatively weak evidence compared with the title.

### Decision

We selected:

> **One vector per article**

This representation produced the strongest overall retrieval performance in
the experiment and avoids the additional complexity of maintaining multiple
vectors per article.

---

## Error Analysis

We manually inspected **6 retrieval errors**.

In all 6 cases, a human reader would have selected the correct problem class.
For example, a complaint describing **"no internet at all"** was matched to an
article describing **intermittent connectivity**.

This suggests that these cases are primarily **retrieval weaknesses rather
than label-overlap problems**.

### Observed / hypothesized causes

Two hypotheses emerged from the manual analysis:

1. **Long embedded text may dilute the signal from the article title.**
2. **A small number of articles appear close to many different complaints.**

There is also an inherent ambiguity within some classes: the complaint alone
may not contain enough information to determine the exact root cause.

This motivates the use of **clarifying questions** when multiple technically
plausible resolutions remain.

---

## Oracle Class: How Much Would a Perfect Classifier Add?

We also evaluated an oracle setting in which retrieval candidates were
restricted to the **true problem class**.

Each class contained **4 candidate articles**.

As a baseline, randomly selecting among the four candidates gives:

| Baseline | R@1 | R@2 | MRR |
|---|---:|---:|---:|
| Random | 0.25 | 0.50 | 0.52 |

The retrieval methods were then evaluated only within the correct class:

| Method | R@1 | R@2 | MRR |
|---|---:|---:|---:|
| Dense | **0.54** | 0.77 | **0.72** |
| BM25 | 0.47 | 0.73 | 0.68 |
| Hybrid | 0.52 | **0.77** | 0.71 |

### Findings

A perfect class prediction would increase R@1 from **0.40** in the best
unrestricted setting to **0.54**.

Therefore, **0.54 represents an approximate upper bound for R@1 under this
oracle-class experiment**, rather than a production result.

Within the correct class:

- Dense retrieval achieved the highest R@1 (**0.54**).
- Hybrid retrieval reached the same R@2 as dense (**0.77**).
- Hybrid did not outperform dense within the known class.

This suggests that the improvement observed for hybrid retrieval in the
unrestricted setting may have come primarily from its **class-level
decision**, rather than from superior fine-grained ranking within an already
correct class.

Even when the correct class is known, the correct article is ranked first
only about half the time. However, it appears in the top two **77%** of the
time.

This provides motivation for a support workflow that can ask **clarifying
questions** when the retrieved evidence does not sufficiently distinguish
between plausible articles.

---

## Experimental Limitations

These results should be interpreted with the following limitations:

- The experiments are based on **synthetic data**.
- The chunking comparison is based on a **single run**.
- The oracle-class experiment uses **53 German tickets**.
- The oracle-class setting assumes a perfect class prediction and therefore
  represents an **upper-bound analysis**, not a deployable configuration.
- Some root causes are genuinely difficult to distinguish from the complaint
  alone.

These experiments are therefore intended primarily to understand the
retrieval behaviour of the system and guide design decisions rather than to
serve as a final production benchmark.

---

---

## Query-Aware Feedback Reranking

After initial retrieval, we added a **query-aware feedback reranking layer** to
refine the ordering of retrieved candidates.

The motivation was that retrieval feedback should not be treated as a static
global score. A document that was useful for one type of complaint may not be
equally useful for another.

The reranker therefore considers the relationship between the **current query**
and previously observed retrieval feedback.

### Retrieval Pipeline

```text
Customer Query
      │
      ▼
┌─────────────────────────┐
│ Initial Retrieval       │
│                         │
│ • Dense similarity      │
│ • BM25                  │
└────────────┬────────────┘
             │
             ▼
      Candidate Pool
             │
             ▼
┌─────────────────────────┐
│ Query-Aware Feedback    │
│ Reranking               │
│                         │
│ • Query similarity      │
│ • Historical feedback   │
│ • Retrieval signals     │
└────────────┬────────────┘
             │
             ▼
      Final Ranking
             │
             ▼
       LLM Context


> class can substantially narrow the retrieval problem, while fine-grained
> article selection remains challenging even within the correct class.
