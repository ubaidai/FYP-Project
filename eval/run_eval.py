"""Retrieval evaluation harness.

Measures the retrieval half of the RAG pipeline against a labelled dataset:
whether the chunk containing the correct answer is actually retrieved, and
whether in-scope and out-of-scope questions are separable by similarity
score (i.e. whether a confidence threshold for abstention is viable).

Usage:
    python eval/run_eval.py                      # evaluate shipped defaults
    python eval/run_eval.py --sweep              # compare chunk sizes
    python eval/run_eval.py --chunk-size 60 --overlap 15
    python eval/run_eval.py --sweep --out eval/results.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.retrieval import (  # noqa: E402
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_TOP_K,
    build_index,
    load_model,
)

RANKS = (1, 3, 5)
SWEEP_SIZES = ((400, 50), (200, 40), (120, 30), (80, 20), (60, 15), (40, 10))


# ── helpers ────────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load_dataset(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def first_hit_rank(results: Sequence[tuple], answer_key: str) -> int | None:
    """1-indexed rank of the first chunk containing the answer, else None."""
    key = normalise(answer_key)
    for rank, (chunk, _score) in enumerate(results, start=1):
        if key in normalise(chunk):
            return rank
    return None


def best_threshold(in_scores: List[float], out_scores: List[float]) -> tuple[float, float]:
    """Threshold maximising correct accept/reject accuracy. Returns (thr, acc)."""
    candidates = sorted(set(in_scores + out_scores))
    best, best_acc = 0.0, 0.0
    total = len(in_scores) + len(out_scores)
    for thr in candidates:
        correct = sum(s >= thr for s in in_scores) + sum(s < thr for s in out_scores)
        acc = correct / total if total else 0.0
        if acc > best_acc:
            best, best_acc = thr, acc
    return best, best_acc


# ── evaluation ─────────────────────────────────────────────────────────────

def evaluate(corpus: str, rows: List[dict], chunk_size: int, overlap: int, top_k: int) -> dict:
    index = build_index(corpus, model=load_model(), chunk_size=chunk_size, overlap=overlap)
    depth = max(top_k, max(RANKS))

    hits = {k: 0 for k in RANKS}
    reciprocal, in_scope_n = 0.0, 0
    in_scores: List[float] = []
    out_scores: List[float] = []
    misses: List[str] = []

    for row in rows:
        results = index.search(row["question"], top_k=depth)
        top_score = results[0][1] if results else 0.0

        if not row["in_scope"]:
            out_scores.append(top_score)
            continue

        in_scope_n += 1
        in_scores.append(top_score)
        rank = first_hit_rank(results, row["answer_key"])
        if rank is None:
            misses.append(row["id"])
            continue
        reciprocal += 1.0 / rank
        for k in RANKS:
            if rank <= k:
                hits[k] += 1

    thr, thr_acc = best_threshold(in_scores, out_scores)
    mean_in = sum(in_scores) / len(in_scores) if in_scores else 0.0
    mean_out = sum(out_scores) / len(out_scores) if out_scores else 0.0

    return {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunks": len(index),
        "queries": in_scope_n,
        "hit_rate": {k: hits[k] / in_scope_n if in_scope_n else 0.0 for k in RANKS},
        "mrr": reciprocal / in_scope_n if in_scope_n else 0.0,
        "mean_score_in_scope": mean_in,
        "mean_score_out_of_scope": mean_out,
        "separation": mean_in - mean_out,
        "threshold": thr,
        "threshold_accuracy": thr_acc,
        "misses": misses,
    }


# ── reporting ──────────────────────────────────────────────────────────────

def format_table(results: List[dict]) -> str:
    head = (
        "| chunk_size | overlap | chunks | hit@1 | hit@3 | hit@5 | MRR | "
        "score in-scope | score out-of-scope | separation |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = "".join(
        "| {chunk_size} | {overlap} | {chunks} | {h1:.1%} | {h3:.1%} | {h5:.1%} | "
        "{mrr:.3f} | {mi:.3f} | {mo:.3f} | {sep:+.3f} |\n".format(
            chunk_size=r["chunk_size"], overlap=r["overlap"], chunks=r["chunks"],
            h1=r["hit_rate"][1], h3=r["hit_rate"][3], h5=r["hit_rate"][5],
            mrr=r["mrr"], mi=r["mean_score_in_scope"],
            mo=r["mean_score_out_of_scope"], sep=r["separation"],
        )
        for r in results
    )
    return head + rows


def print_detail(r: dict) -> None:
    print(f"  chunks produced      : {r['chunks']}")
    print(f"  in-scope queries     : {r['queries']}")
    for k in RANKS:
        print(f"  hit-rate@{k}          : {r['hit_rate'][k]:.1%}")
    print(f"  MRR                  : {r['mrr']:.3f}")
    print(f"  mean score in-scope  : {r['mean_score_in_scope']:.3f}")
    print(f"  mean score off-topic : {r['mean_score_out_of_scope']:.3f}")
    print(f"  separation           : {r['separation']:+.3f}")
    print(f"  best threshold       : {r['threshold']:.3f} "
          f"(accept/reject accuracy {r['threshold_accuracy']:.1%})")
    if r["misses"]:
        print(f"  missed queries       : {', '.join(r['misses'])}")


# ── entry point ────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate RAG retrieval quality.")
    ap.add_argument("--corpus", type=Path, default=ROOT / "data" / "sample_faq.txt")
    ap.add_argument("--dataset", type=Path, default=ROOT / "eval" / "dataset.jsonl")
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--sweep", action="store_true", help="compare chunking strategies")
    ap.add_argument("--out", type=Path, help="write a markdown report to this path")
    args = ap.parse_args()

    corpus = args.corpus.read_text(encoding="utf-8")
    rows = load_dataset(args.dataset)
    in_scope = sum(r["in_scope"] for r in rows)
    print(f"corpus  : {args.corpus.name} ({len(corpus.split())} words)")
    print(f"dataset : {len(rows)} queries ({in_scope} in-scope, {len(rows) - in_scope} off-topic)\n")

    configs = SWEEP_SIZES if args.sweep else ((args.chunk_size, args.overlap),)
    results = []
    for size, overlap in configs:
        print(f"chunk_size={size} overlap={overlap}")
        r = evaluate(corpus, rows, size, overlap, args.top_k)
        print_detail(r)
        print()
        results.append(r)

    if len(results) > 1:
        print(format_table(results))
        best = max(results, key=lambda r: (r["hit_rate"][3], r["mrr"]))
        base = results[0]
        print(f"best: chunk_size={best['chunk_size']} overlap={best['overlap']} "
              f"-> hit@3 {base['hit_rate'][3]:.1%} => {best['hit_rate'][3]:.1%}")

    if args.out:
        args.out.write_text(format_table(results), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
