"""
search.py — Semantic and hybrid search over knowledge_chunks.

Usage (interactive test):
    python search.py "how much does it cost to jump"
    python search.py "birthday party" --category "Birthday Parties" --mode hybrid
    python search.py "GROUPBOOKING25" --mode hybrid
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

import psycopg2.extras

import config
import embedding as emb
from models import ChunkRecord, SearchResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_SEMANTIC_SQL = """
SELECT
    id, category, subcategory, location, question, answer, tags,
    1 - (embedding <=> %(query_vec)s::vector) AS similarity
FROM knowledge_chunks
WHERE (%(category)s IS NULL OR category = %(category)s)
ORDER BY embedding <=> %(query_vec)s::vector
LIMIT %(top_k)s;
"""

# Hybrid: weighted sum of semantic similarity (0.7) and tag keyword overlap (0.3).
# Tag keyword match: any tag that is a sub-string of the lower-cased query gets a hit.
_HYBRID_SQL = """
WITH base AS (
    SELECT
        id, category, subcategory, location, question, answer, tags,
        1 - (embedding <=> %(query_vec)s::vector) AS semantic_score,
        (
            SELECT COUNT(*)::float / GREATEST(array_length(tags, 1), 1)
            FROM unnest(tags) AS t
            WHERE lower(%(query)s) LIKE '%%' || lower(t) || '%%'
               OR lower(t) LIKE '%%' || lower(%(query)s) || '%%'
        ) AS keyword_score
    FROM knowledge_chunks
    WHERE (%(category)s IS NULL OR category = %(category)s)
)
SELECT
    id, category, subcategory, location, question, answer, tags,
    (%(semantic_w)s * semantic_score + %(keyword_w)s * keyword_score) AS similarity
FROM base
ORDER BY similarity DESC
LIMIT %(top_k)s;
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_chunk(row: dict) -> ChunkRecord:
    return ChunkRecord(
        id=row["id"],
        category=row["category"],
        subcategory=row["subcategory"],
        location=row["location"],
        question=row["question"],
        answer=row["answer"],
        tags=list(row["tags"]),
    )


def _fetch(sql: str, params: dict) -> list[SearchResult]:
    pool = config.get_db_pool()
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [SearchResult(chunk=_row_to_chunk(r), similarity_score=float(r["similarity"])) for r in rows]
    finally:
        pool.putconn(conn)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def semantic_search(
    query: str,
    category: Optional[str] = None,
    top_k: int = config.DEFAULT_TOP_K,
) -> list[SearchResult]:
    """
    Pure vector-cosine search.

    Args:
        query: Natural-language question from the user.
        category: Optional filter (e.g. "FAQ", "Pricing", "Birthday Parties").
        top_k: Number of results to return.

    Returns:
        List of SearchResult ordered by descending similarity.
    """
    query_vec = emb.embed_text(query, input_type="query")
    return _fetch(
        _SEMANTIC_SQL,
        {
            "query_vec": query_vec,
            "category": category,
            "top_k": top_k,
        },
    )


def hybrid_search(
    query: str,
    category: Optional[str] = None,
    top_k: int = config.DEFAULT_TOP_K,
) -> list[SearchResult]:
    """
    Hybrid search: semantic (0.7) + tag keyword overlap (0.3).

    Better for short promo codes and exact keywords (e.g. "GROUPBOOKING25").

    Args:
        query: Natural-language or keyword query.
        category: Optional category filter.
        top_k: Number of results to return.

    Returns:
        List of SearchResult ordered by descending combined score.
    """
    query_vec = emb.embed_text(query, input_type="query")
    return _fetch(
        _HYBRID_SQL,
        {
            "query_vec": query_vec,
            "query": query,
            "category": category,
            "top_k": top_k,
            "semantic_w": config.HYBRID_SEMANTIC_WEIGHT,
            "keyword_w": config.HYBRID_KEYWORD_WEIGHT,
        },
    )


# ---------------------------------------------------------------------------
# CLI test harness
# ---------------------------------------------------------------------------


def _print_results(results: list[SearchResult]) -> None:
    if not results:
        print("No results found.")
        return
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r.chunk.id}  score={r.similarity_score:.4f}")
        print(f"    Category : {r.chunk.category} > {r.chunk.subcategory}")
        print(f"    Question : {r.chunk.question}")
        print(f"    Answer   : {r.chunk.answer[:200]}{'…' if len(r.chunk.answer) > 200 else ''}")
        print(f"    Tags     : {', '.join(r.chunk.tags)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search AeroSports knowledge base")
    parser.add_argument("query", nargs="?", default=None, help="Search query")
    parser.add_argument("--category", default=None, help="Filter by category")
    parser.add_argument("--mode", choices=["semantic", "hybrid"], default="hybrid", help="Search mode (default: hybrid)")
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    args = parser.parse_args()

    if args.query is None:
        # Interactive mode
        print("AeroSports Search  (Ctrl-C to exit)\n")
        while True:
            try:
                query = input("Query> ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not query:
                continue
            fn = hybrid_search if args.mode == "hybrid" else semantic_search
            _print_results(fn(query, category=args.category, top_k=args.top_k))
    else:
        fn = hybrid_search if args.mode == "hybrid" else semantic_search
        _print_results(fn(args.query, category=args.category, top_k=args.top_k))

    config.close_db_pool()


if __name__ == "__main__":
    main()
