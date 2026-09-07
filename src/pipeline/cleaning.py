"""Cleaning stage: turns the raw, column-mapped DataFrame from ingest.py into
validated `ShipmentRecord`s plus a structured data-quality report.

Handles the three messiness categories called out in the case study:
  - inconsistent date formats (and outright garbage date strings)
  - missing required/optional fields
  - unstandardized location names (abbreviations, misspellings, casing)
plus exact-duplicate row removal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import pandas as pd
from dateutil import parser as dateutil_parser
from rapidfuzz import fuzz, process

from src.pipeline.schema import (
    CANONICAL_LOCATIONS,
    LOCATION_ALIASES,
    REQUIRED_FIELDS,
    ShipmentRecord,
    ShipmentStatus,
)

# Tokens legacy systems used in place of a real date. Treated as "missing",
# not as a parse error worth surfacing individually.
_GARBAGE_DATE_TOKENS = {"", "n/a", "na", "unknown", "tbd", "0000-00-00", "none", "null"}

_LOCATION_FUZZY_THRESHOLD = 78

_STATUS_MAP = {
    "delivered": ShipmentStatus.DELIVERED,
    "in_transit": ShipmentStatus.IN_TRANSIT,
    "in transit": ShipmentStatus.IN_TRANSIT,
    "delayed": ShipmentStatus.DELAYED,
    "cancelled": ShipmentStatus.CANCELLED,
    "canceled": ShipmentStatus.CANCELLED,
}


def parse_flexible_date(raw: str) -> Optional[date]:
    """Parse a date string in one of many legacy formats. Returns None for
    blank/garbage/unparseable values instead of raising."""
    if raw is None:
        return None
    text = str(raw).strip()
    if text.lower() in _GARBAGE_DATE_TOKENS:
        return None
    try:
        # dayfirst=False matches the US-centric formats in our source systems;
        # dateutil still correctly handles unambiguous formats like "24 Apr 2025".
        return dateutil_parser.parse(text, dayfirst=False, fuzzy=False).date()
    except (ValueError, OverflowError):
        return None


def standardize_location(raw: str) -> tuple[str, bool]:
    """Map a dirty location string to a canonical name.

    Returns (canonical_or_original, matched) where `matched` is False when we
    could not confidently resolve the value (caller should flag it).
    """
    if raw is None:
        return "", False
    text = str(raw).strip()
    if not text:
        return "", False

    normalized = text.lower()
    if normalized in LOCATION_ALIASES:
        return LOCATION_ALIASES[normalized], True

    for canonical in CANONICAL_LOCATIONS:
        if normalized == canonical.lower():
            return canonical, True

    match = process.extractOne(
        text, CANONICAL_LOCATIONS, scorer=fuzz.WRatio, score_cutoff=_LOCATION_FUZZY_THRESHOLD
    )
    if match:
        canonical_name, _score, _idx = match
        return canonical_name, True

    return text, False


def _normalize_status(raw: str) -> ShipmentStatus:
    return _STATUS_MAP.get(str(raw).strip().lower(), ShipmentStatus.UNKNOWN)


@dataclass
class CleaningReport:
    total_input_rows: int = 0
    duplicate_rows_removed: int = 0
    rejected_rows: list[dict] = field(default_factory=list)
    unmatched_locations: dict[str, int] = field(default_factory=dict)
    missing_optional_field_counts: dict[str, int] = field(default_factory=dict)
    accepted_rows: int = 0

    def to_dict(self) -> dict:
        return {
            "total_input_rows": self.total_input_rows,
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "accepted_rows": self.accepted_rows,
            "rejected_row_count": len(self.rejected_rows),
            "rejected_rows_sample": self.rejected_rows[:20],
            "unmatched_locations": self.unmatched_locations,
            "missing_optional_field_counts": self.missing_optional_field_counts,
        }


def clean_shipments(raw_df: pd.DataFrame) -> tuple[list[ShipmentRecord], CleaningReport]:
    report = CleaningReport(total_input_rows=len(raw_df))

    before = len(raw_df)
    raw_df = raw_df.drop_duplicates()
    report.duplicate_rows_removed = before - len(raw_df)

    records: list[ShipmentRecord] = []

    for _, row in raw_df.iterrows():
        row_dict = row.to_dict()
        source_file = row_dict.get("source_file", "unknown")

        # --- required field presence check (before type coercion) ---
        missing_required = [
            f for f in REQUIRED_FIELDS if not str(row_dict.get(f, "")).strip()
        ]

        ship_date = parse_flexible_date(row_dict.get("ship_date"))
        if ship_date is None and "ship_date" not in missing_required:
            missing_required.append("ship_date")

        if missing_required:
            report.rejected_rows.append(
                {
                    "shipment_id": row_dict.get("shipment_id") or "<missing>",
                    "source_file": source_file,
                    "reason": f"missing/unparseable required field(s): {missing_required}",
                }
            )
            continue

        origin, origin_matched = standardize_location(row_dict.get("origin"))
        destination, dest_matched = standardize_location(row_dict.get("destination"))
        if not origin_matched:
            report.unmatched_locations[origin] = report.unmatched_locations.get(origin, 0) + 1
        if not dest_matched:
            report.unmatched_locations[destination] = report.unmatched_locations.get(destination, 0) + 1

        expected_delivery = parse_flexible_date(row_dict.get("expected_delivery_date"))
        actual_delivery = parse_flexible_date(row_dict.get("actual_delivery_date"))

        for opt_field, value in (
            ("route_id", row_dict.get("route_id")),
            ("carrier", row_dict.get("carrier")),
            ("expected_delivery_date", row_dict.get("expected_delivery_date") if expected_delivery else None),
            ("actual_delivery_date", row_dict.get("actual_delivery_date") if actual_delivery else None),
        ):
            if not value or not str(value).strip():
                report.missing_optional_field_counts[opt_field] = (
                    report.missing_optional_field_counts.get(opt_field, 0) + 1
                )

        status = _normalize_status(row_dict.get("status"))

        delay_days: Optional[int] = None
        is_delayed = False
        if expected_delivery and actual_delivery:
            delay_days = (actual_delivery - expected_delivery).days
            is_delayed = delay_days > 0
        elif status == ShipmentStatus.DELAYED:
            is_delayed = True

        try:
            record = ShipmentRecord(
                shipment_id=row_dict["shipment_id"],
                route_id=row_dict.get("route_id") or None,
                origin=origin,
                destination=destination,
                carrier=row_dict.get("carrier") or None,
                ship_date=ship_date,
                expected_delivery_date=expected_delivery,
                actual_delivery_date=actual_delivery,
                status=status,
                delay_days=delay_days,
                is_delayed=is_delayed,
                source_file=source_file,
            )
        except Exception as exc:  # pydantic ValidationError or similar
            report.rejected_rows.append(
                {
                    "shipment_id": row_dict.get("shipment_id") or "<missing>",
                    "source_file": source_file,
                    "reason": f"validation error: {exc}",
                }
            )
            continue

        records.append(record)

    report.accepted_rows = len(records)
    return records, report
