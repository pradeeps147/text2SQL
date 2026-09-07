"""Executes validated SQL against a read-only DuckDB connection, with a
timeout and an LLM self-healing retry loop for execution errors.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from src.nlq.answer_synthesis import fallback_answer, synthesize_answer
from src.nlq.guardrails import validate_sql
from src.nlq.llm_client import LLMClient, build_llm_client
from src.nlq.retrieval import format_retrieved_context, retrieve_context
from src.nlq.text_to_sql import generate_sql
from src.pipeline.load import get_schema_description

ALLOWED_TABLES = {"shipments"}
ALLOWED_COLUMNS = {
    "shipment_id",
    "route_id",
    "origin",
    "destination",
    "carrier",
    "ship_date",
    "expected_delivery_date",
    "actual_delivery_date",
    "status",
    "delay_days",
    "is_delayed",
    "source_file",
    "ingested_at",
}

MAX_RETRIES = 2
QUERY_TIMEOUT_SECONDS = 15
logger = logging.getLogger("tenarai.nlq")


@dataclass
class QueryResult:
    question: str
    sql: str = ""
    dataframe: Optional[pd.DataFrame] = None
    answer: str = ""
    sql_explanation: str = ""
    retrieved_context: list[dict] = field(default_factory=list)
    synthesis_error: str = ""
    error: str = ""
    attempts: list[dict] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.error == "" and self.dataframe is not None


def _run_with_timeout(con: duckdb.DuckDBPyConnection, sql: str, timeout_seconds: int) -> pd.DataFrame:
    outcome: dict = {}

    def _work() -> None:
        try:
            outcome["df"] = con.execute(sql).fetchdf()
        except Exception as exc:  # noqa: BLE001 - propagated to caller below
            outcome["error"] = exc

    thread = threading.Thread(target=_work, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        con.interrupt()
        thread.join(2)
        raise TimeoutError(f"Query exceeded {timeout_seconds}s timeout and was cancelled.")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["df"]


def answer_question(
    question: str,
    db_path: Path,
    client: Optional[LLMClient] = None,
    max_retries: int = MAX_RETRIES,
    timeout_seconds: int = QUERY_TIMEOUT_SECONDS,
    request_id: str | None = None,
) -> QueryResult:
    """Translate a natural-language question into SQL, validate it, execute
    it, and self-heal (re-prompt the LLM with the error) up to `max_retries`
    times if generation or execution fails."""
    client = client or build_llm_client()
    result = QueryResult(question=question)

    # Read-only connection: even if a bug slipped a mutating statement past
    # validate_sql, DuckDB itself refuses writes on this connection.
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        schema_description = get_schema_description(con)
        retrieved = retrieve_context(question)
        result.retrieved_context = [item.to_dict() for item in retrieved]
        retrieved_text = format_retrieved_context(retrieved)
        logger.info(
            "Retrieved Text-to-SQL grounding context",
            extra={
                "event": "rag.retrieved",
                "request_id": request_id,
                "details": {"documents": [item.document_id for item in retrieved]},
            },
        )
        error_feedback: Optional[str] = None

        for attempt in range(1, max_retries + 2):  # 1 initial try + N retries
            generation = generate_sql(
                client,
                question,
                schema_description,
                retrieved_context=retrieved_text,
                error_feedback=error_feedback,
            )

            if not generation.sql:
                result.error = generation.explanation or "Model declined to generate SQL for this question."
                result.attempts.append({"attempt": attempt, "sql": "", "error": result.error})
                logger.warning(
                    "LLM did not generate SQL",
                    extra={
                        "event": "chat.sql_declined",
                        "request_id": request_id,
                        "details": {"attempt": attempt, "error": result.error},
                    },
                )
                break

            validation = validate_sql(generation.sql, ALLOWED_TABLES, ALLOWED_COLUMNS)
            if not validation.is_valid:
                result.error = validation.error
                result.attempts.append({"attempt": attempt, "sql": generation.sql, "error": validation.error})
                error_feedback = validation.error
                logger.warning(
                    "Generated SQL failed validation",
                    extra={
                        "event": "chat.sql_rejected",
                        "request_id": request_id,
                        "details": {"attempt": attempt, "error": validation.error},
                    },
                )
                continue

            try:
                df = _run_with_timeout(con, validation.sql, timeout_seconds)
            except Exception as exc:  # noqa: BLE001 - fed back to the model as feedback
                result.error = str(exc)
                result.attempts.append({"attempt": attempt, "sql": validation.sql, "error": str(exc)})
                error_feedback = str(exc)
                logger.warning(
                    "DuckDB query execution failed",
                    extra={
                        "event": "chat.sql_failed",
                        "request_id": request_id,
                        "details": {"attempt": attempt, "error": str(exc)},
                    },
                )
                continue

            result.sql = validation.sql
            result.sql_explanation = generation.explanation
            result.dataframe = df
            result.error = ""
            result.attempts.append({"attempt": attempt, "sql": validation.sql, "error": None})
            try:
                result.answer = synthesize_answer(client, question, validation.sql, df)
                if not result.answer:
                    raise ValueError("The synthesis model returned an empty answer.")
            except Exception as exc:  # noqa: BLE001 - table result remains useful
                result.synthesis_error = str(exc)
                result.answer = fallback_answer(df)
                logger.warning(
                    "Natural-language answer synthesis failed; using deterministic fallback",
                    extra={
                        "event": "chat.synthesis_failed",
                        "request_id": request_id,
                        "details": {"error": str(exc)},
                    },
                )
            logger.info(
                "Chat query completed",
                extra={
                    "event": "chat.completed",
                    "request_id": request_id,
                    "details": {"attempt": attempt, "rows": len(df), "columns": list(df.columns)},
                },
            )
            break
    finally:
        con.close()

    return result
