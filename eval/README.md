# Retrieval Evaluation

Measures the retrieval half of the pipeline. Generation quality is not scored here — if the correct chunk never reaches the LLM, no amount of prompting recovers it, so retrieval is the first thing worth measuring.

The harness imports `core.retrieval`, the same module `utils/vector_store.py` uses, so these numbers describe the code path the app actually runs.

## Running it

```bash
python eval/run_eval.py                     # shipped defaults
python eval/run_eval.py --sweep             # compare chunking strategies
python eval/run_eval.py --chunk-size 200 --overlap 40
```

## Dataset

`dataset.jsonl` — 55 labelled queries against `data/sample_faq.txt`:

- **45 in-scope**, covering 15 topics with 3 paraphrases each. Questions are written the way a customer would ask ("Where is my parcel right now?"), deliberately avoiding the FAQ's own wording, so the test measures semantic retrieval rather than keyword overlap.
- **10 off-topic**, plausible-but-unanswerable questions ("Do you ship internationally to Canada?") used to test whether the system can tell what it doesn't know.

A query counts as a hit when a retrieved chunk contains the labelled answer span.

## Metrics

| Metric | Meaning |
|---|---|
| hit-rate@k | Share of in-scope queries whose answer appears in the top *k* chunks |
| MRR | Mean reciprocal rank of the first correct chunk — rewards ranking it first |
| separation | Mean top-1 similarity for in-scope minus off-topic queries |
| threshold accuracy | Best achievable accept/reject accuracy using a similarity cutoff |

## Results

Corpus: 442 words. 45 in-scope queries, 10 off-topic.

| chunk_size | overlap | chunks | hit@1 | hit@3 | hit@5 | MRR | score in-scope | score off-topic | separation |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **400** | **50** | **2** | **66.7%** | 100.0% | 100.0% | 0.833 | 0.301 | 0.156 | +0.145 |
| 200 | 40 | 3 | 77.8% | 100.0% | 100.0% | **0.870** | 0.322 | 0.171 | +0.151 |
| 120 | 30 | 5 | 75.6% | 93.3% | 100.0% | 0.843 | 0.360 | 0.193 | +0.168 |
| 80 | 20 | 8 | 73.3% | 95.6% | 100.0% | 0.843 | 0.383 | 0.204 | +0.179 |
| 60 | 15 | 10 | 73.3% | 93.3% | 95.6% | 0.831 | 0.420 | 0.213 | +0.207 |
| 40 | 10 | 15 | 62.2% | 84.4% | 100.0% | 0.764 | 0.443 | 0.223 | **+0.220** |

Bold row is the shipped default.

## What this actually shows

**1. The default chunk size makes retrieval close to a no-op.** At `chunk_size=400`, a 442-word knowledge base becomes **2 chunks**. Retrieving the top 4 returns the entire corpus every time, so hit@3 and hit@5 of 100% mean nothing — there is nothing to rank. The only informative number in that row is hit@1 (66.7%), the worst of any configuration tested.

**2. Score separation improves monotonically as chunks get smaller** — +0.145 at 400 words up to +0.220 at 40. Smaller chunks are less diluted, so an off-topic question is less likely to partially match. This is the most robust trend in the table because it holds across every step.

**3. But the corpus is too small for the hit-rate comparison to be conclusive.** With 45 queries, the gap between 66.7% and 77.8% is five queries — inside the noise band for this sample size. Treat `chunk_size=200` as *directionally* better, not proven better. A larger corpus is required before claiming a tuned win.

**4. There is no abstention gate at all.** `app.py` sends the top 4 chunks to the LLM regardless of similarity, then infers "unanswered" by string-matching the reply for `"don't have information"`. The measured best threshold tops out at **87.3% accept/reject accuracy**, so roughly 1 in 8 questions would still be misrouted even with a perfectly tuned cutoff — and today there is no cutoff, so off-topic questions are answered from irrelevant context and rely entirely on the LLM to refuse.

## Next

- Replace the 442-word sample with a corpus of a few thousand words so chunking differences become measurable rather than suggestive
- Add the confidence gate using `retrieve_with_scores`, and re-measure abstention precision/recall separately rather than as a single accuracy number
- Score answer quality, not just retrieval, with an LLM judge over the same dataset
- Report variance across repeated runs before treating any hit-rate difference as real
