"""Small, deterministic query plans; no generated facts or model dependency."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .keyword_search import tokenize


_INTENT_WORDS = frozenset({
    "construction", "sequence", "instructions", "materials", "dimensions",
    "assembly", "diagram", "diagrams", "illustration", "layout", "procedure",
})
_VISUAL = re.compile(
    r"\b(?:diagram|diagrams|illustration|illustrations|visual|visually|visible|"
    r"image|images|picture|pictures|photo|photograph|appearance|layout)\b|"
    r"\b(?:look like|where .+ appears?|chart .*(?:look|show|appearance))\b", re.I
)
_SEQUENCE = re.compile(r"\b(?:construction|sequence|procedure|workflow|step names|steps|order of)\b", re.I)
_PRODUCT = re.compile(r"\b(?:which|what) (?:digital )?product\b", re.I)


@dataclass(frozen=True)
class QueryPart:
    query: str
    intent: str


@dataclass(frozen=True)
class QueryPlan:
    normalized_query: str
    parts: tuple[QueryPart, ...]
    corrections: tuple[tuple[str, str], ...]
    names_only: bool

    @property
    def visual(self) -> bool:
        return any(part.intent == "visual" for part in self.parts)


def _edit_distance(left: str, right: str) -> int:
    """Bounded-size optimal-string-alignment distance, including transpositions."""
    rows = [list(range(len(right) + 1))]
    for i, a in enumerate(left, 1):
        row = [i]
        for j, b in enumerate(right, 1):
            cost = min(row[-1] + 1, rows[-1][j] + 1, rows[-1][j - 1] + (a != b))
            if i > 1 and j > 1 and a == right[j - 2] and left[i - 2] == b:
                cost = min(cost, rows[-2][j - 2] + 1)
            row.append(cost)
        rows.append(row)
    return rows[-1][-1]


def plan_query(
    query: str,
    *,
    vocabulary: Iterable[str],
    product_titles: Iterable[str],
) -> QueryPlan:
    """Correct only unambiguous long title/intent typos, then split question clauses.

    Known corpus words, numbers, identifiers, paths, and quoted strings are kept.
    The limited correction vocabulary avoids turning unknown personal facts into
    unrelated corpus terms. Unrecognized words remain available to semantic search.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    known = set(vocabulary)
    correction_words = (_INTENT_WORDS & known) | {
        term for title in product_titles for term in tokenize(title) if term.isalpha()
    }
    corrections: list[tuple[str, str]] = []

    def correct(match: re.Match[str]) -> str:
        word = match.group(0)
        lower = word.casefold()
        if not word.isalpha() or len(word) < 6 or len(word) > 24 or lower in known:
            return word
        candidates = difflib.get_close_matches(lower, sorted(correction_words), n=5, cutoff=0.78)
        ranked = sorted((_edit_distance(lower, candidate), candidate) for candidate in candidates)
        if not ranked or ranked[0][0] > 2 or (len(ranked) > 1 and ranked[0][0] == ranked[1][0]):
            return word
        replacement = ranked[0][1]
        corrections.append((word, replacement))
        return replacement

    # Match protected spans first; correction applies only to plain standalone words.
    normalized = re.sub(
        r'"[^"\n]*"|\b[\w]+(?:[./\\:_-][\w]+)+\b|\b[^\W\d_]+\b',
        correct, query.strip(), flags=re.UNICODE,
    )
    names_only = bool(re.search(r"\bnames? only\b|\bonly (?:the )?(?:step )?names?\b", normalized, re.I))
    clauses = re.split(r"\s+(?:and|also)\s+(?=(?:what|which|how|where|tell|list)\b)|[?;]+", normalized, flags=re.I)
    parts: list[QueryPart] = []
    for clause in clauses[:4]:
        if not clause.strip():
            continue
        intent = "visual" if _VISUAL.search(clause) else (
            "sequence" if _SEQUENCE.search(clause) else "product" if _PRODUCT.search(clause) else "text"
        )
        clean = re.sub(r"\btell me\b|\b(?:the )?(?:step )?names? only\b|\bonly (?:the )?names?\b", " ", clause, flags=re.I)
        clean = " ".join(clean.split())
        # A sequence retrieval synonym specifies the information need, never an answer.
        if intent == "sequence" and "sequence" not in tokenize(clean):
            clean += " sequence"
        if tokenize(clean):
            parts.append(QueryPart(clean, intent))
    if not parts:
        parts = [QueryPart(normalized, "text")]
    return QueryPlan(normalized, tuple(parts), tuple(dict.fromkeys(corrections)), names_only)


def prioritize_sequence(results: list[Any], part: QueryPart) -> list[Any]:
    """Prefer explicit written sequence structure within an already retrieved pool."""
    if part.intent != "sequence":
        return results

    def explicit_sequence(result: Any) -> bool:
        return result.chunk_type == "text" and bool(
            re.search(r"\b(?:sequence|order|workflow)\b[^\n]{0,60}:?", result.text, re.I)
            and ("→" in result.text or re.search(r"(?m)^\s*\d+[.)]\s", result.text))
        )

    return sorted(results, key=lambda item: not explicit_sequence(item))


def combine_query_results(rankings: list[list[Any]], *, top_k: int) -> list[Any]:
    """Round-robin ranks preserve each subquestion's strongest evidence and scores.

    Unlike reinterpreting subquery ranks as semantic/BM25 ranks, this retains the
    original retriever diagnostics. Duplicate chunks and identical page passages
    appear once; source order within each passage remains untouched.
    """
    output: list[Any] = []
    seen_ids: set[str] = set()
    seen_passages: set[tuple[Any, ...]] = set()
    for rank in range(max((len(items) for items in rankings), default=0)):
        for items in rankings:
            if rank >= len(items):
                continue
            item = items[rank]
            signature = (item.document_id, tuple(item.source_pages), " ".join(item.text.casefold().split()))
            if item.chunk_id in seen_ids or signature in seen_passages:
                continue
            output.append(item)
            seen_ids.add(item.chunk_id)
            seen_passages.add(signature)
            if len(output) >= top_k:
                return output
    return output
