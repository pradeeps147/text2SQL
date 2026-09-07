import duckdb
import pytest

from src.pipeline.ingest import load_uploaded_csvs
from src.services import dataset_service
from src.services.dataset_service import UploadPayload, process_uploaded_csvs, validate_uploads


CSV_A = b"ShipmentRef,Route,From,To,CarrierName,PickupDate,ETA,DeliveredOn,Status\nS1,R1,NYC,HOU,Swift,2025-01-01,2025-01-05,2025-01-06,delayed\n"
CSV_B = b"id,route_id,origin,destination,carrier,ship_date,expected_delivery_date,actual_delivery_date,status\nS2,R2,Chicago,Miami,Arrow,2025-02-01,2025-02-05,2025-02-05,delivered\n"


def test_multiple_uploaded_csvs_are_reconciled():
    result = load_uploaded_csvs([("legacy-a.csv", CSV_A), ("legacy-b.csv", CSV_B)])

    assert result.files_loaded == ["legacy-a.csv", "legacy-b.csv"]
    assert len(result.raw_df) == 2
    assert set(result.raw_df["shipment_id"]) == {"S1", "S2"}


def test_upload_validation_rejects_non_csv_and_duplicate_names():
    with pytest.raises(ValueError, match="not a CSV"):
        validate_uploads([UploadPayload("notes.txt", b"hello")])

    with pytest.raises(ValueError, match="Duplicate filename"):
        validate_uploads([UploadPayload("a.csv", CSV_A), UploadPayload("A.csv", CSV_B)])


def test_upload_pipeline_atomically_builds_queryable_duckdb(tmp_path, monkeypatch):
    db_path = tmp_path / "active.duckdb"
    report_path = tmp_path / "quality.json"
    monkeypatch.setattr(dataset_service, "UPLOAD_ROOT", tmp_path / "uploads")

    result = process_uploaded_csvs(
        [UploadPayload("a.csv", CSV_A), UploadPayload("b.csv", CSV_B)],
        db_path=db_path,
        report_path=report_path,
    )

    assert result["accepted_rows"] == 2
    assert result["files_loaded"] == ["a.csv", "b.csv"]
    assert report_path.exists()
    assert (dataset_service.UPLOAD_ROOT / result["batch_id"]).exists()

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        assert con.execute("SELECT COUNT(*) FROM shipments").fetchone()[0] == 2
    finally:
        con.close()
