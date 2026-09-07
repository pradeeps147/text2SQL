from datetime import date

import pandas as pd

from src.pipeline.cleaning import (
    clean_shipments,
    parse_flexible_date,
    standardize_location,
)
from src.pipeline.schema import ShipmentStatus


def _row(**overrides) -> dict:
    base = {
        "shipment_id": "SHP000001",
        "route_id": "NEW-HOU",
        "origin": "New York",
        "destination": "Houston",
        "carrier": "Swift Freight",
        "ship_date": "2025-01-01",
        "expected_delivery_date": "2025-01-05",
        "actual_delivery_date": "2025-01-07",
        "status": "DELAYED",
        "source_file": "test.csv",
    }
    base.update(overrides)
    return base


class TestParseFlexibleDate:
    def test_iso_format(self):
        assert parse_flexible_date("2025-03-14") == date(2025, 3, 14)

    def test_us_slash_format(self):
        assert parse_flexible_date("03/14/2025") == date(2025, 3, 14)

    def test_day_month_dash_format(self):
        assert parse_flexible_date("14-03-2025") == date(2025, 3, 14)

    def test_month_name_format(self):
        assert parse_flexible_date("Mar 14, 2025") == date(2025, 3, 14)

    def test_garbage_tokens_return_none(self):
        for token in ["", "N/A", "unknown", "TBD", "0000-00-00", None]:
            assert parse_flexible_date(token) is None

    def test_unparseable_string_returns_none(self):
        assert parse_flexible_date("not-a-date-at-all!!") is None


class TestStandardizeLocation:
    def test_alias_lookup(self):
        assert standardize_location("NYC") == ("New York", True)
        assert standardize_location("N.York") == ("New York", True)

    def test_case_insensitive_exact_match(self):
        assert standardize_location("chicago") == ("Chicago", True)

    def test_fuzzy_misspelling(self):
        name, matched = standardize_location("Los Angles")
        assert matched is True
        assert name == "Los Angeles"

    def test_unmatched_returns_original_flagged_false(self):
        name, matched = standardize_location("Springfield")
        assert matched is False
        assert name == "Springfield"

    def test_blank_returns_unmatched(self):
        assert standardize_location("") == ("", False)


class TestCleanShipments:
    def test_valid_row_is_accepted(self):
        df = pd.DataFrame([_row()])
        records, report = clean_shipments(df)
        assert report.accepted_rows == 1
        assert len(records) == 1
        assert records[0].status == ShipmentStatus.DELAYED.value
        assert records[0].delay_days == 2
        assert records[0].is_delayed is True

    def test_missing_required_field_is_rejected(self):
        df = pd.DataFrame([_row(origin="")])
        records, report = clean_shipments(df)
        assert len(records) == 0
        assert report.accepted_rows == 0
        assert len(report.rejected_rows) == 1

    def test_unparseable_ship_date_is_rejected(self):
        df = pd.DataFrame([_row(ship_date="not-a-date")])
        records, report = clean_shipments(df)
        assert len(records) == 0
        assert len(report.rejected_rows) == 1

    def test_missing_optional_field_is_flagged_not_rejected(self):
        df = pd.DataFrame([_row(carrier="")])
        records, report = clean_shipments(df)
        assert len(records) == 1
        assert report.missing_optional_field_counts.get("carrier") == 1

    def test_exact_duplicate_rows_are_removed(self):
        df = pd.DataFrame([_row(), _row()])
        records, report = clean_shipments(df)
        assert report.duplicate_rows_removed == 1
        assert len(records) == 1

    def test_location_standardization_applied(self):
        df = pd.DataFrame([_row(origin="NYC", destination="HOU")])
        records, _ = clean_shipments(df)
        assert records[0].origin == "New York"
        assert records[0].destination == "Houston"

    def test_in_transit_has_no_delay_computed(self):
        df = pd.DataFrame(
            [_row(status="IN_TRANSIT", actual_delivery_date="")]
        )
        records, _ = clean_shipments(df)
        assert records[0].delay_days is None
        assert records[0].is_delayed is False
