"""Ingestion: discover raw CSV files and reconcile their differing column
layouts into a single raw DataFrame with canonical column names.

Each legacy export system uses different header names for the same concept
(e.g. "ShipmentRef", "id", "shipment_id" all mean the same thing). Rather than
hardcoding per-file parsers, we normalize headers and map them through a
synonym table, then union the results. Unrecognized columns are kept aside
(not silently discarded) so they can be reported.
"""

from __future__ import annotations

import re
from io import BytesIO
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# normalized header -> canonical raw column name.
# Normalization = lowercase, strip everything but letters/digits.
COLUMN_SYNONYMS: dict[str, str] = {
    # shipment_id
    "shipmentid": "shipment_id",
    "shipmentref": "shipment_id",
    "id": "shipment_id",
    # route_id
    "routeid": "route_id",
    "route": "route_id",
    # origin
    "origin": "origin",
    "from": "origin",
    "origincity": "origin",
    # destination
    "destination": "destination",
    "to": "destination",
    "destcity": "destination",
    # carrier
    "carrier": "carrier",
    "carriername": "carrier",
    # ship_date
    "shipdate": "ship_date",
    "pickupdate": "ship_date",
    "shipdt": "ship_date",
    # expected_delivery_date
    "expecteddeliverydate": "expected_delivery_date",
    "eta": "expected_delivery_date",
    # actual_delivery_date
    "actualdeliverydate": "actual_delivery_date",
    "deliveredon": "actual_delivery_date",
    "deliverydt": "actual_delivery_date",
    # status
    "status": "status",
}

RAW_COLUMNS: tuple[str, ...] = (
    "shipment_id",
    "route_id",
    "origin",
    "destination",
    "carrier",
    "ship_date",
    "expected_delivery_date",
    "actual_delivery_date",
    "status",
)


def _normalize_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.strip().lower())


@dataclass
class IngestResult:
    raw_df: pd.DataFrame
    unmapped_columns: dict[str, list[str]] = field(default_factory=dict)
    files_loaded: list[str] = field(default_factory=list)


def _normalize_dataframe(df: pd.DataFrame, source_name: str) -> tuple[pd.DataFrame, list[str]]:
    """Map one source DataFrame into the canonical raw shipment layout."""
    rename_map: dict[str, str] = {}
    unmapped: list[str] = []
    for col in df.columns:
        canonical = COLUMN_SYNONYMS.get(_normalize_header(col))
        if canonical:
            rename_map[col] = canonical
        else:
            unmapped.append(col)
    df = df.rename(columns=rename_map)

    # Ensure every canonical column exists, even if this source never has it.
    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df[list(RAW_COLUMNS)].copy()
    df["source_file"] = source_name
    return df, unmapped


def _load_one_csv(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return _normalize_dataframe(df, path.name)


def _load_csv_bytes(filename: str, content: bytes) -> tuple[pd.DataFrame, list[str]]:
    """Load a CSV received from an API/UI upload without writing it first."""
    try:
        df = pd.read_csv(BytesIO(content), dtype=str, keep_default_na=False)
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Could not parse '{filename}' as CSV: {exc}") from exc
    return _normalize_dataframe(df, filename)


def load_raw_csvs(raw_dir: Path) -> IngestResult:
    """Load and column-map every CSV file under `raw_dir`."""
    csv_paths = sorted(raw_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    frames: list[pd.DataFrame] = []
    unmapped_by_file: dict[str, list[str]] = {}
    files_loaded: list[str] = []

    for path in csv_paths:
        df, unmapped = _load_one_csv(path)
        frames.append(df)
        files_loaded.append(path.name)
        if unmapped:
            unmapped_by_file[path.name] = unmapped

    combined = pd.concat(frames, ignore_index=True)
    return IngestResult(raw_df=combined, unmapped_columns=unmapped_by_file, files_loaded=files_loaded)


def load_uploaded_csvs(files: list[tuple[str, bytes]]) -> IngestResult:
    """Load multiple uploaded CSV payloads into one reconciled DataFrame.

    ``files`` is deliberately framework-neutral so it can be used by the API,
    tests, and future batch workers without depending on FastAPI or Streamlit.
    """
    if not files:
        raise ValueError("At least one CSV file is required.")

    frames: list[pd.DataFrame] = []
    unmapped_by_file: dict[str, list[str]] = {}
    files_loaded: list[str] = []

    for filename, content in files:
        df, unmapped = _load_csv_bytes(filename, content)
        frames.append(df)
        files_loaded.append(filename)
        if unmapped:
            unmapped_by_file[filename] = unmapped

    return IngestResult(
        raw_df=pd.concat(frames, ignore_index=True),
        unmapped_columns=unmapped_by_file,
        files_loaded=files_loaded,
    )
