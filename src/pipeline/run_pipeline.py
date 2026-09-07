"""End-to-end pipeline orchestration: ingest raw CSVs -> clean -> load into DuckDB.

Run:
    python -m src.pipeline.run_pipeline
"""

from __future__ import annotations

from pathlib import Path

from src.pipeline.cleaning import clean_shipments
from src.pipeline.ingest import load_raw_csvs
from src.pipeline.load import DEFAULT_DB_PATH, DEFAULT_REPORT_PATH, load_to_duckdb, write_quality_report

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"


def run_pipeline(raw_dir: Path = RAW_DIR, db_path: Path = DEFAULT_DB_PATH, report_path: Path = DEFAULT_REPORT_PATH):
    ingest_result = load_raw_csvs(raw_dir)
    records, report = clean_shipments(ingest_result.raw_df)
    con = load_to_duckdb(records, db_path=db_path)
    write_quality_report(report, path=report_path)
    return con, report, ingest_result


if __name__ == "__main__":
    con, report, ingest_result = run_pipeline()
    print(f"Files loaded: {ingest_result.files_loaded}")
    if ingest_result.unmapped_columns:
        print(f"Unmapped columns by file: {ingest_result.unmapped_columns}")
    print(f"Total input rows: {report.total_input_rows}")
    print(f"Duplicates removed: {report.duplicate_rows_removed}")
    print(f"Accepted rows: {report.accepted_rows}")
    print(f"Rejected rows: {len(report.rejected_rows)}")
    print(f"Unmatched locations: {report.unmatched_locations}")
    row_count = con.execute("SELECT COUNT(*) FROM shipments").fetchone()[0]
    print(f"Rows in DuckDB shipments table: {row_count}")
