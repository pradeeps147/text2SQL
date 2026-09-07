"""Load cleaned shipment records into a DuckDB analytical store."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from src.pipeline.cleaning import CleaningReport
from src.pipeline.schema import ShipmentRecord

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "shipments.duckdb"
DEFAULT_REPORT_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "data_quality_report.json"
)


def records_to_dataframe(records: list[ShipmentRecord]) -> pd.DataFrame:
    rows = [r.model_dump() for r in records]
    df = pd.DataFrame(rows)
    for col in ("ship_date", "expected_delivery_date", "actual_delivery_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.date
    return df


def load_to_duckdb(
    records: list[ShipmentRecord],
    db_path: Path = DEFAULT_DB_PATH,
) -> duckdb.DuckDBPyConnection:
    """(Re)create the `shipments` table in DuckDB from cleaned records."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    df = records_to_dataframe(records)

    con = duckdb.connect(str(db_path))
    con.execute("DROP TABLE IF EXISTS shipments")
    con.register("shipments_df", df)
    con.execute(
        """
        CREATE TABLE shipments AS
        SELECT
            shipment_id,
            route_id,
            origin,
            destination,
            carrier,
            CAST(ship_date AS DATE) AS ship_date,
            CAST(expected_delivery_date AS DATE) AS expected_delivery_date,
            CAST(actual_delivery_date AS DATE) AS actual_delivery_date,
            status,
            delay_days,
            is_delayed,
            source_file,
            ingested_at
        FROM shipments_df
        """
    )
    con.unregister("shipments_df")
    return con


def write_quality_report(report: CleaningReport, path: Path = DEFAULT_REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, default=str))


def get_schema_description(con: duckdb.DuckDBPyConnection, table: str = "shipments") -> str:
    """Human/LLM-readable description of a table's columns and types, plus a
    couple of sample rows, for grounding Text-to-SQL prompts."""
    cols = con.execute(
        "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position",
        [table],
    ).fetchall()
    col_lines = "\n".join(f"  - {name}: {dtype}" for name, dtype in cols)

    sample = con.execute(f"SELECT * FROM {table} LIMIT 3").fetchdf()
    sample_str = sample.to_string(index=False)

    return f"Table: {table}\nColumns:\n{col_lines}\n\nSample rows:\n{sample_str}"
