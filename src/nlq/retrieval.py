"""Lightweight RAG retrieval for shipment-domain Text-to-SQL grounding.

The corpus is intentionally small and auditable: a business glossary, metric
definitions, and SQL guidance. Retrieval selects only the entries relevant to
the user's question and combines them with the live DuckDB schema.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class KnowledgeDocument:
    document_id: str
    title: str
    content: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class RetrievedContext:
    document_id: str
    title: str
    content: str
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


KNOWLEDGE_BASE: tuple[KnowledgeDocument, ...] = (
    KnowledgeDocument(
        "metric.delay",
        "Delay and on-time metrics",
        "A shipment is delayed when is_delayed is true. Delay days is actual_delivery_date minus expected_delivery_date. "
        "Delay rate is delayed shipments divided by all shipments in the selected group. On-time rate is one minus delay rate.",
        ("delay", "delayed", "late", "on time", "punctual", "delay rate", "delay days"),
    ),
    KnowledgeDocument(
        "dimension.route",
        "Routes and lanes",
        "route_id identifies a logistics lane. origin and destination are standardized city names. "
        "Questions about lanes, corridors, or origin-to-destination movement should group by route_id and may include origin and destination.",
        ("route", "lane", "corridor", "origin", "destination", "from", "to"),
    ),
    KnowledgeDocument(
        "dimension.carrier",
        "Carrier performance",
        "carrier is the logistics provider. Carrier performance can be measured using shipment count, average delay_days, and delay rate. "
        "Null carrier values represent unavailable source data and should not be silently relabeled.",
        ("carrier", "provider", "transporter", "freight", "performance"),
    ),
    KnowledgeDocument(
        "dimension.status",
        "Shipment status",
        "status is one of DELIVERED, IN_TRANSIT, DELAYED, CANCELLED, or UNKNOWN. "
        "Use is_delayed for delivery-performance calculations because it also reflects date-derived delays.",
        ("status", "delivered", "transit", "cancelled", "canceled", "unknown"),
    ),
    KnowledgeDocument(
        "time.relative",
        "Relative time periods",
        "The data may be historical. Resolve last month, last quarter, and last year relative to MAX(ship_date), not the current system date. "
        "Use half-open date ranges so the end boundary is excluded.",
        ("last", "recent", "month", "quarter", "year", "week", "date", "period", "trend"),
    ),
    KnowledgeDocument(
        "quality.provenance",
        "Source and data quality",
        "source_file identifies the uploaded CSV that contributed a row. ingested_at records when the clean record was created. "
        "Missing optional values remain null; rejected rows are not loaded into shipments.",
        ("source", "file", "uploaded", "quality", "missing", "rejected", "ingested"),
    ),
)

_STOP_TERMS = {
    "a",
    "all",
    "and",
    "are",
    "by",
    "for",
    "how",
    "in",
    "is",
    "of",
    "or",
    "shipment",
    "shipments",
    "show",
    "the",
    "to",
    "was",
    "were",
    "what",
    "which",
    "with",
}


def _terms(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9_]+", text.lower())
    terms = set(words) - _STOP_TERMS
    for word in words:
        if len(word) > 4 and word.endswith("s") and word[:-1] not in _STOP_TERMS:
            terms.add(word[:-1])
        if len(word) > 5 and word.endswith("ed") and word[:-2] not in _STOP_TERMS:
            terms.add(word[:-2])
    return terms


def retrieve_context(question: str, top_k: int = 3) -> list[RetrievedContext]:
    """Rank glossary entries using deterministic lexical/phrase relevance."""
    normalized_question = question.lower()
    question_terms = _terms(question)
    ranked: list[RetrievedContext] = []

    for document in KNOWLEDGE_BASE:
        keyword_score = 0.0
        for keyword in document.keywords:
            if " " in keyword and keyword in normalized_question:
                keyword_score += 3.0
            elif keyword in question_terms:
                keyword_score += 2.0

        content_overlap = len(question_terms & _terms(f"{document.title} {document.content}"))
        score = keyword_score + (content_overlap * 0.25)
        if score > 0:
            ranked.append(
                RetrievedContext(
                    document_id=document.document_id,
                    title=document.title,
                    content=document.content,
                    score=round(score, 2),
                )
            )

    ranked.sort(key=lambda item: (-item.score, item.document_id))
    return ranked[:top_k]


def format_retrieved_context(items: list[RetrievedContext]) -> str:
    return "\n".join(f"- [{item.document_id}] {item.title}: {item.content}" for item in items)


def describe_rag_pipeline() -> dict:
    """Machine-readable explanation exposed by the backend and UI."""
    return {
        "strategy": "hybrid schema + lexical business-glossary retrieval",
        "steps": [
            "Read the live DuckDB schema and representative rows.",
            "Rank the business glossary against the user's question.",
            "Inject the top glossary entries and schema into the SQL-generation prompt.",
            "Validate generated SQL with table/column allowlists and enforce a row limit.",
            "Execute through a read-only DuckDB connection with a timeout.",
            "Send the bounded result table back to the LLM for a plain-language answer.",
        ],
        "knowledge_documents": len(KNOWLEDGE_BASE),
        "retrieval_top_k": 3,
        "note": "This compact corpus uses deterministic lexical retrieval; it does not require an embedding database.",
    }
