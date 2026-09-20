#!/usr/bin/env python3
"""
Benchmark script for evaluating chunking strategies on ecommerce policy documents.
Reads .md files, parses frontmatter, chunks content, indexes in EmbeddingStore,
and runs 5 benchmark queries with search_with_filter.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from src.agent import KnowledgeBaseAgent
from src.chunking import (
    ChunkingStrategyComparator,
    FixedSizeChunker,
    HeadingChunker,
    RecursiveChunker,
    SentenceChunker,
)
from src.embeddings import _mock_embed
from src.models import Document
from src.store import EmbeddingStore


# =============================================================================
# CONFIGURATION — Change only this line to switch chunking strategy
# =============================================================================
# CHUNKER = FixedSizeChunker(chunk_size=500, overlap=50)
# CHUNKER = SentenceChunker(max_sentences_per_chunk=3)
# CHUNKER = RecursiveChunker(chunk_size=500)
CHUNKER = HeadingChunker(chunk_size=500)
# CHUNKER = HeadingChunker(chunk_size=500)  # Your custom strategy here

DATA_DIR = Path("data/ecommerce")
OUTPUT_CSV = Path("bench_results.csv")
CACHE_FILE = Path(".embedding_cache.json")

# 5 Benchmark queries with gold answers (from REPORT_NHOM.md)
QUERIES = [
    {
        "id": "Q1",
        "query": "What is the return policy for buyers?",
        "gold_answer": "Buyers can return items within the specified period (e.g., 7-14 days) with original packaging. Refund issued after inspection.",
        "filter": None,
        "expected_doc_ids": ["cotopaxi-returns", "shopee-buyer-return-refund-rules", "return-refund-policy"],
    },
    {
        "id": "Q2",
        "query": "What are the conditions for a valid return?",
        "gold_answer": "Items must be unused, with original tags and packaging. Return within policy period. Some items like underwear/swimwear may be excluded.",
        "filter": None,
        "expected_doc_ids": ["cotopaxi-returns", "qoatea-doi-tra", "shopee-buyer-return-refund-rules"],
    },
    {
        "id": "Q3",
        "query": "How long does a refund take to process?",
        "gold_answer": "Refunds typically processed within 5-14 business days after return received and inspected.",
        "filter": None,
        "expected_doc_ids": ["cotopaxi-returns", "shopee-buyer-return-refund-rules", "return-refund-policy"],
    },
    {
        "id": "Q4",
        "query": "What is the shipping policy for domestic orders?",
        "gold_answer": "Free shipping on orders over threshold. Standard delivery 3-7 business days. Expedited options available.",
        "filter": None,
        "expected_doc_ids": ["cotopaxi-shipping", "shopee-buyer-return-refund-rules"],
    },
    {
        "id": "Q5",
        "query": "What is the seller warranty policy?",
        "gold_answer": "Sellers must provide warranty for eligible products. Coverage period and claim process defined in seller agreement.",
        "filter": {"audience": "seller"},
        "expected_doc_ids": ["seller-warranty-policy"],
    },
]


def parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    """Parse YAML frontmatter from markdown content."""
    if not content.startswith("---"):
        return {}, content
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content
    fm_text = parts[1].strip()
    body = parts[2].strip()
    metadata = {}
    for line in fm_text.split("\n"):
        if ":" in line:
            key, val = line.split(":", 1)
            val = val.strip().strip('"')
            # Strip inline comments
            if "#" in val:
                val = val.split("#")[0].strip()
            metadata[key.strip()] = val
    return metadata, body


def load_documents(data_dir: Path, chunker) -> list[Document]:
    """Load all .md files, parse frontmatter, chunk body, create Documents."""
    docs: list[Document] = []
    for md_file in sorted(data_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        frontmatter, body = parse_frontmatter(content)
        if not body:
            continue

        chunks = chunker.chunk(body)
        for i, chunk in enumerate(chunks):
            doc_id = f"{md_file.stem}#{i}"
            metadata = {**frontmatter, "doc_id": md_file.stem, "chunk_index": str(i)}
            docs.append(Document(id=doc_id, content=chunk, metadata=metadata))
    return docs


def load_cache() -> dict[str, list[float]]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_cache(cache: dict[str, list[float]]) -> None:
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def get_embedding(text: str, cache: dict[str, list[float]]) -> list[float]:
    """Get embedding with caching by content hash."""
    key = hashlib.md5(text.encode()).hexdigest()
    if key in cache:
        return cache[key]
    embedding = _mock_embed(text)
    cache[key] = embedding
    return embedding


def run_benchmark() -> None:
    print(f"=== Benchmark: {CHUNKER.__class__.__name__} ===")
    print(f"Data directory: {DATA_DIR}")

    cache = load_cache()

    docs = load_documents(DATA_DIR, CHUNKER)
    print(f"Loaded {len(docs)} chunks from {len(list(DATA_DIR.glob('*.md')))} files")

    store = EmbeddingStore(collection_name="bench", embedding_fn=lambda t: get_embedding(t, cache))
    store.add_documents(docs)
    print(f"Indexed {store.get_collection_size()} documents in EmbeddingStore")

    save_cache(cache)

    results_rows = []
    for q in QUERIES:
        filter_desc = f" | filter: {q['filter']}" if q["filter"] else ""
        print(f"\n{q['id']}: {q['query']}{filter_desc}")

        results = store.search_with_filter(q["query"], top_k=3, metadata_filter=q["filter"])

        for rank, r in enumerate(results, 1):
            is_gold = r["metadata"].get("doc_id") in q["expected_doc_ids"]
            marker = " OK" if is_gold else ""
            print("  {}. score={:.3f} doc_id={} chunk={}{}".format(rank, r['score'], r['metadata'].get('doc_id'), r['id'], marker))

            results_rows.append({
                "query_id": q["id"],
                "query": q["query"],
                "filter": str(q["filter"]) if q["filter"] else "",
                "rank": rank,
                "doc_id": r["metadata"].get("doc_id", ""),
                "chunk_id": r["id"],
                "score": f"{r['score']:.3f}",
                "is_gold": "1" if is_gold else "0",
                "content_preview": r["content"][:100].replace("\n", " "),
            })

    # Write CSV results
    if results_rows:
        with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results_rows[0].keys()))
            writer.writeheader()
            writer.writerows(results_rows)
        print(f"\nResults written to {OUTPUT_CSV}")

    print("\n=== Baseline Analysis (ChunkingStrategyComparator) ===")
    sample_text = "\n\n".join([d.content for d in docs[:3]])
    comparator = ChunkingStrategyComparator()
    comparison = comparator.compare(sample_text, chunk_size=500)
    for strategy, stats in comparison.items():
        print(f"  {strategy}: {stats['count']} chunks, avg_len={stats['avg_length']:.0f}")


if __name__ == "__main__":
    run_benchmark()