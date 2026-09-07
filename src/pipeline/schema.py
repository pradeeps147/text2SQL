"""Canonical data contract for cleaned shipment records.

This is the single source of truth for what a "clean" shipment row looks like.
Both the cleaning stage (src/pipeline/cleaning.py) and the DuckDB load stage
(src/pipeline/load.py) validate against this schema, and the Text-to-SQL layer
(src/nlq/text_to_sql.py) uses it to describe the table to the LLM.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# Reference ("master data") list of canonical location names. In a real
# enterprise system this would come from a locations/geo dimension table; here
# it doubles as the fuzzy-matching target list for cleaning.py.
CANONICAL_LOCATIONS: list[str] = [
    "New York",
    "Los Angeles",
    "Chicago",
    "Houston",
    "Atlanta",
    "Seattle",
    "Miami",
    "Denver",
]

# Known aliases/abbreviations seen in legacy exports -> canonical name.
# Checked before falling back to fuzzy matching.
LOCATION_ALIASES: dict[str, str] = {
    "nyc": "New York",
    "n.york": "New York",
    "new york city": "New York",
    "la": "Los Angeles",
    "l.a.": "Los Angeles",
    "chi": "Chicago",
    "hou": "Houston",
    "atl": "Atlanta",
    "sea": "Seattle",
    "mia": "Miami",
    "den": "Denver",
}


class ShipmentStatus(str, Enum):
    DELIVERED = "DELIVERED"
    IN_TRANSIT = "IN_TRANSIT"
    DELAYED = "DELAYED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


# Fields that MUST be present and valid for a row to be loaded. Rows missing
# any of these are rejected (not silently dropped) and logged in the data
# quality report.
REQUIRED_FIELDS: tuple[str, ...] = ("shipment_id", "origin", "destination", "ship_date")

# Fields that may legitimately be missing (e.g. a shipment still in transit
# has no actual_delivery_date). Missing optional fields are flagged, not rejected.
OPTIONAL_FIELDS: tuple[str, ...] = ("route_id", "carrier", "expected_delivery_date", "actual_delivery_date")


class ShipmentRecord(BaseModel):
    shipment_id: str
    route_id: Optional[str] = None
    origin: str
    destination: str
    carrier: Optional[str] = None
    ship_date: date
    expected_delivery_date: Optional[date] = None
    actual_delivery_date: Optional[date] = None
    status: ShipmentStatus = ShipmentStatus.UNKNOWN
    delay_days: Optional[int] = None
    is_delayed: bool = False
    source_file: str
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("shipment_id", "origin", "destination")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be blank")
        return v.strip()

    model_config = {"use_enum_values": True}
