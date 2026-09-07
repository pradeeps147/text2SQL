"""FastAPI backend for Tenarai ingestion, chat, RAG, and observability."""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import uuid
from collections import defaultdict, deque
from time import monotonic, perf_counter
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.nlq.executor import answer_question
from src.nlq.llm_client import DEFAULT_MODEL, DEFAULT_PROVIDER, LLMConnectionError, build_llm_client
from src.nlq.retrieval import describe_rag_pipeline
from src.observability.logging import configure_logging, get_recent_logs
from src.pipeline.load import DEFAULT_DB_PATH
from src.services.dataset_service import MAX_FILE_BYTES, UploadPayload, get_dataset_summary, process_uploaded_csvs

logger = configure_logging()
ADMIN_KEY = os.environ.get("TENARAI_ADMIN_KEY", "")
ALLOW_MODEL_OVERRIDE = os.environ.get("ALLOW_MODEL_OVERRIDE", "true").lower() in {"1", "true", "yes"}
CHAT_LIMIT_PER_MINUTE = max(1, int(os.environ.get("TENARAI_CHAT_LIMIT_PER_MINUTE", "30")))
_CHAT_REQUESTS: dict[str, deque[float]] = defaultdict(deque)
_CHAT_LIMIT_LOCK = threading.Lock()

app = FastAPI(
    title="Tenarai logistics intelligence API",
    description="Multi-CSV ingestion and retrieval-grounded Text-to-SQL chat over DuckDB.",
    version="2.0.0",
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    model: str | None = Field(default=None, min_length=1, max_length=200)


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    sql: str
    sql_explanation: str
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    attempts: list[dict]
    retrieved_context: list[dict]
    synthesis_warning: str | None = None


def require_admin(x_tenarai_admin_key: Annotated[str | None, Header()] = None) -> None:
    """Protect operational data when an admin key is configured."""
    if ADMIN_KEY and (
        not x_tenarai_admin_key or not secrets.compare_digest(x_tenarai_admin_key, ADMIN_KEY)
    ):
        raise HTTPException(status_code=401, detail="A valid Tenarai admin key is required.")


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id
    started = perf_counter()
    if request.url.path == "/api/v1/chat":
        forwarded = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        client_key = forwarded or (request.client.host if request.client else "unknown")
        now = monotonic()
        with _CHAT_LIMIT_LOCK:
            recent = _CHAT_REQUESTS[client_key]
            while recent and now - recent[0] >= 60:
                recent.popleft()
            if len(recent) >= CHAT_LIMIT_PER_MINUTE:
                logger.warning(
                    "Public chat rate limit exceeded",
                    extra={
                        "event": "chat.rate_limited",
                        "request_id": request_id,
                        "details": {"limit": CHAT_LIMIT_PER_MINUTE},
                    },
                )
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Public chat rate limit exceeded. Try again in one minute."},
                    headers={"X-Request-ID": request_id, "Retry-After": "60"},
                )
            recent.append(now)
    try:
        response = await call_next(request)
    except Exception:  # noqa: BLE001 - preserve framework exception handling
        logger.exception(
            "Unhandled API request failure",
            extra={"event": "http.failed", "request_id": request_id, "details": {"path": request.url.path}},
        )
        raise
    duration_ms = round((perf_counter() - started) * 1000, 1)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        f"{request.method} {request.url.path} {response.status_code}",
        extra={
            "event": "http.completed",
            "request_id": request_id,
            "duration_ms": duration_ms,
            "details": {"method": request.method, "path": request.url.path, "status": response.status_code},
        },
    )
    return response


@app.exception_handler(ValueError)
async def value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/")
def root() -> dict:
    return {
        "service": "Tenarai logistics intelligence API",
        "version": app.version,
        "docs": "/docs",
    }


@app.get("/api/v1/health")
def health() -> dict:
    dataset = get_dataset_summary()
    return {
        "status": "ok",
        "dataset_ready": dataset["ready"],
        "version": app.version,
        "llm_provider": DEFAULT_PROVIDER,
        "model": DEFAULT_MODEL,
        "allow_model_override": ALLOW_MODEL_OVERRIDE,
        "admin_protected": bool(ADMIN_KEY),
        "chat_limit_per_minute": CHAT_LIMIT_PER_MINUTE,
    }


@app.get("/api/v1/dataset")
def dataset_summary() -> dict:
    return get_dataset_summary()


@app.post("/api/v1/datasets/upload")
async def upload_dataset(
    files: Annotated[list[UploadFile], File(description="One or more shipment CSV files")],
    _admin: Annotated[None, Depends(require_admin)],
) -> dict:
    payloads: list[UploadPayload] = []
    for upload in files:
        content = await upload.read(MAX_FILE_BYTES + 1)
        payloads.append(UploadPayload(filename=upload.filename or "", content=content))
        await upload.close()
    return process_uploaded_csvs(payloads)


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    if not DEFAULT_DB_PATH.exists():
        raise HTTPException(status_code=409, detail="No active dataset. Upload CSV files before asking questions.")

    request_id = request.state.request_id
    selected_model = payload.model if ALLOW_MODEL_OVERRIDE and payload.model else DEFAULT_MODEL
    client = build_llm_client(model=selected_model)
    try:
        client.check_connection()
    except LLMConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    logger.info(
        "Chat request started",
        extra={
            "event": "chat.started",
            "request_id": request_id,
            "details": {"provider": DEFAULT_PROVIDER, "model": selected_model},
        },
    )
    result = answer_question(
        payload.question,
        db_path=DEFAULT_DB_PATH,
        client=client,
        request_id=request_id,
    )
    if not result.succeeded:
        logger.warning(
            "Chat request could not be answered",
            extra={"event": "chat.failed", "request_id": request_id, "details": {"error": result.error}},
        )
        raise HTTPException(status_code=422, detail={"message": result.error, "attempts": result.attempts})

    rows = json.loads(result.dataframe.to_json(orient="records", date_format="iso"))
    return ChatResponse(
        request_id=request_id,
        answer=result.answer,
        sql=result.sql,
        sql_explanation=result.sql_explanation,
        columns=[str(column) for column in result.dataframe.columns],
        rows=rows,
        row_count=len(result.dataframe),
        attempts=result.attempts,
        retrieved_context=result.retrieved_context,
        synthesis_warning=result.synthesis_error or None,
    )


@app.get("/api/v1/rag")
def rag_details() -> dict:
    return describe_rag_pipeline()


@app.get("/api/v1/logs")
def logs(
    _admin: Annotated[None, Depends(require_admin)],
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
    level: Annotated[str | None, Query(pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")] = None,
) -> dict:
    records = get_recent_logs(limit=limit, level=level)
    return {"records": records, "count": len(records)}
