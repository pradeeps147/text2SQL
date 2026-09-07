"""Transactional multi-CSV ingestion for the active Tenarai dataset."""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import duckdb

from src.pipeline.cleaning import clean_shipments
from src.pipeline.ingest import IngestResult, load_uploaded_csvs
from src.pipeline.load import DEFAULT_DB_PATH, DEFAULT_REPORT_PATH, load_to_duckdb, write_quality_report

MAX_UPLOAD_FILES = 20
MAX_FILE_BYTES = 25 * 1024 * 1024
UPLOAD_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "uploads"

logger = logging.getLogger("tenarai.ingestion")
_INGESTION_LOCK = threading.Lock()


@dataclass(frozen=True)
class UploadPayload:
    filename: str
    content: bytes


def _safe_filename(filename: str) -> str:
    name = filename.replace("\\", "/").split("/")[-1].strip()
    if not name or name in {".", ".."}:
        raise ValueError("Every upload must have a valid filename.")
    if not name.lower().endswith(".csv"):
        raise ValueError(f"'{name}' is not a CSV file.")
    return name


def validate_uploads(files: list[UploadPayload]) -> list[UploadPayload]:
    if not files:
        raise ValueError("Upload at least one CSV file.")
    if len(files) > MAX_UPLOAD_FILES:
        raise ValueError(f"A maximum of {MAX_UPLOAD_FILES} CSV files can be uploaded at once.")

    validated: list[UploadPayload] = []
    seen: set[str] = set()
    for upload in files:
        name = _safe_filename(upload.filename)
        normalized_name = name.lower()
        if normalized_name in seen:
            raise ValueError(f"Duplicate filename in this batch: '{name}'.")
        if not upload.content:
            raise ValueError(f"'{name}' is empty.")
        if len(upload.content) > MAX_FILE_BYTES:
            raise ValueError(f"'{name}' exceeds the 25 MB per-file limit.")
        seen.add(normalized_name)
        validated.append(UploadPayload(filename=name, content=upload.content))
    return validated


def _persist_upload_batch(files: list[UploadPayload], batch_id: str) -> Path:
    batch_dir = UPLOAD_ROOT / batch_id
    batch_dir.mkdir(parents=True, exist_ok=False)
    for upload in files:
        (batch_dir / upload.filename).write_bytes(upload.content)
    return batch_dir


def process_uploaded_csvs(
    files: list[UploadPayload],
    db_path: Path = DEFAULT_DB_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict:
    """Clean a complete upload batch and atomically replace the active DB."""
    validated = validate_uploads(files)
    started = perf_counter()
    batch_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    logger.info(
        "CSV upload batch received",
        extra={
            "event": "ingestion.received",
            "details": {"batch_id": batch_id, "files": [item.filename for item in validated]},
        },
    )

    with _INGESTION_LOCK:
        ingest_result: IngestResult = load_uploaded_csvs(
            [(item.filename, item.content) for item in validated]
        )
        records, report = clean_shipments(ingest_result.raw_df)
        if not records:
            raise ValueError("The upload contained no valid shipment rows; the active dataset was not changed.")

        db_path.parent.mkdir(parents=True, exist_ok=True)
        staging_db = db_path.with_name(f".{db_path.stem}-{uuid.uuid4().hex}.duckdb")
        staging_report = report_path.with_name(f".{report_path.stem}-{uuid.uuid4().hex}.json")
        try:
            con = load_to_duckdb(records, db_path=staging_db)
            con.close()
            write_quality_report(report, path=staging_report)
            os.replace(staging_db, db_path)
            os.replace(staging_report, report_path)
            batch_dir = _persist_upload_batch(validated, batch_id)
        finally:
            staging_db.unlink(missing_ok=True)
            staging_report.unlink(missing_ok=True)

    duration_ms = round((perf_counter() - started) * 1000, 1)
    response = {
        "batch_id": batch_id,
        "files_loaded": ingest_result.files_loaded,
        "total_input_rows": report.total_input_rows,
        "accepted_rows": report.accepted_rows,
        "rejected_rows": len(report.rejected_rows),
        "duplicate_rows_removed": report.duplicate_rows_removed,
        "unmatched_locations": report.unmatched_locations,
        "unmapped_columns": ingest_result.unmapped_columns,
        "missing_optional_field_counts": report.missing_optional_field_counts,
        "storage_key": f"data/uploads/{batch_dir.name}",
        "duration_ms": duration_ms,
    }
    logger.info(
        "CSV upload batch processed",
        extra={"event": "ingestion.completed", "duration_ms": duration_ms, "details": response},
    )
    return response


def get_dataset_summary(db_path: Path = DEFAULT_DB_PATH) -> dict:
    if not db_path.exists():
        return {"ready": False, "row_count": 0, "message": "No active DuckDB dataset."}

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        row = con.execute(
            """
            SELECT
                COUNT(*) AS row_count,
                COUNT(DISTINCT source_file) AS source_count,
                MIN(ship_date) AS min_ship_date,
                MAX(ship_date) AS max_ship_date,
                ROUND(AVG(CASE WHEN is_delayed THEN 1.0 ELSE 0.0 END) * 100, 1) AS delay_rate_pct
            FROM shipments
            """
        ).fetchone()
    finally:
        con.close()

    quality_report = None
    if DEFAULT_REPORT_PATH.exists():
        try:
            quality_report = json.loads(DEFAULT_REPORT_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            quality_report = None

    return {
        "ready": True,
        "row_count": row[0],
        "source_count": row[1],
        "min_ship_date": row[2].isoformat() if row[2] else None,
        "max_ship_date": row[3].isoformat() if row[3] else None,
        "delay_rate_pct": row[4],
        "quality_report": quality_report,
    }
