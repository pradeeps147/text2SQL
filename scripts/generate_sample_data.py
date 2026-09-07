"""Generate synthetic, deliberately messy shipment CSVs for the pipeline demo.

Simulates 3 legacy export files from different regional systems, each with:
- different column names / column subsets for the same concepts
- inconsistent date formats (and some unparseable garbage)
- missing required and optional fields
- misspelled / aliased / inconsistently-cased location names
- duplicate rows

Run:
    python scripts/generate_sample_data.py
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

random.seed(42)

# Canonical cities and the "dirty" aliases legacy systems recorded for them.
CITY_ALIASES: dict[str, list[str]] = {
    "New York": ["New York", "new york", "NYC", "New York City", "N.York"],
    "Los Angeles": ["Los Angeles", "LA", "los angeles", "L.A.", "Los Angles"],
    "Chicago": ["Chicago", "chicago", "CHI", "Chcago"],
    "Houston": ["Houston", "houston", "HOU", "Huoston"],
    "Atlanta": ["Atlanta", "atlanta", "ATL", "Atalanta"],
    "Seattle": ["Seattle", "seattle", "SEA", "Seatle"],
    "Miami": ["Miami", "miami", "MIA", "Maimi"],
    "Denver": ["Denver", "denver", "DEN", "Denvor"],
}
CITIES = list(CITY_ALIASES.keys())

CARRIERS = ["Swift Freight", "BlueLine Logistics", "Pioneer Cargo", "TransHaul Co"]
STATUSES = ["DELIVERED", "IN_TRANSIT", "DELAYED", "CANCELLED"]

DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%b %d, %Y", "%d %b %Y"]


def _random_date(start: date, end: date) -> date:
    delta_days = (end - start).days
    return start + timedelta(days=random.randint(0, delta_days))


def _fmt_date(d: date | None) -> str:
    if d is None:
        return ""
    fmt = random.choice(DATE_FORMATS)
    return d.strftime(fmt)


def _dirty_city(city: str) -> str:
    return random.choice(CITY_ALIASES[city])


def _maybe_blank(value: str, blank_rate: float) -> str:
    return "" if random.random() < blank_rate else value


def _make_row(row_id: int) -> dict:
    origin, destination = random.sample(CITIES, 2)
    ship_date = _random_date(date(2025, 1, 1), date(2025, 9, 1))
    transit_days = random.randint(1, 10)
    expected_delivery = ship_date + timedelta(days=transit_days)

    status = random.choices(STATUSES, weights=[0.55, 0.15, 0.25, 0.05])[0]
    if status == "DELIVERED":
        drift = random.randint(-1, 4)
        actual_delivery = expected_delivery + timedelta(days=max(drift, 0))
    elif status == "DELAYED":
        actual_delivery = expected_delivery + timedelta(days=random.randint(1, 7))
    else:
        actual_delivery = None

    return {
        "shipment_id": f"SHP{row_id:06d}",
        "route_id": f"{origin[:3].upper()}-{destination[:3].upper()}",
        "origin": _dirty_city(origin),
        "destination": _dirty_city(destination),
        "carrier": random.choice(CARRIERS),
        "ship_date": _fmt_date(ship_date),
        "expected_delivery_date": _fmt_date(expected_delivery),
        "actual_delivery_date": _fmt_date(actual_delivery),
        "status": status,
    }


def write_system_a(rows: list[dict], path: Path) -> None:
    """Legacy system A: full column set, ISO-ish dates, occasional blanks."""
    fieldnames = [
        "shipment_id",
        "route_id",
        "origin",
        "destination",
        "carrier",
        "ship_date",
        "expected_delivery_date",
        "actual_delivery_date",
        "status",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["carrier"] = _maybe_blank(out["carrier"], 0.05)
            out["actual_delivery_date"] = _maybe_blank(out["actual_delivery_date"], 0.05)
            writer.writerow(out)


def write_system_b(rows: list[dict], path: Path) -> None:
    """Legacy system B: renamed columns, no route_id, higher missing rate."""
    fieldnames = [
        "ShipmentRef",
        "From",
        "To",
        "Carrier Name",
        "PickupDate",
        "ETA",
        "DeliveredOn",
        "Status",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "ShipmentRef": row["shipment_id"],
                    "From": row["origin"],
                    "To": row["destination"],
                    "Carrier Name": _maybe_blank(row["carrier"], 0.1),
                    "PickupDate": row["ship_date"],
                    "ETA": row["expected_delivery_date"],
                    "DeliveredOn": _maybe_blank(row["actual_delivery_date"], 0.2),
                    "Status": row["status"].lower(),
                }
            )


def write_system_c(rows: list[dict], path: Path) -> None:
    """Legacy system C: minimal columns, some garbage dates, missing ids."""
    fieldnames = [
        "id",
        "origin_city",
        "dest_city",
        "ship_dt",
        "delivery_dt",
        "status",
    ]
    garbage_dates = ["N/A", "unknown", "0000-00-00", "TBD", ""]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            ship_dt = row["ship_date"]
            delivery_dt = row["actual_delivery_date"] or row["expected_delivery_date"]
            if random.random() < 0.08:
                delivery_dt = random.choice(garbage_dates)
            writer.writerow(
                {
                    "id": _maybe_blank(row["shipment_id"], 0.03),
                    "origin_city": row["origin"],
                    "dest_city": row["destination"],
                    "ship_dt": ship_dt,
                    "delivery_dt": delivery_dt,
                    "status": row["status"],
                }
            )


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = [_make_row(i) for i in range(1, 601)]
    # Inject exact duplicate rows to simulate re-exported/overlapping extracts.
    all_rows += random.sample(all_rows, 15)
    random.shuffle(all_rows)

    third = len(all_rows) // 3
    rows_a, rows_b, rows_c = (
        all_rows[:third],
        all_rows[third : 2 * third],
        all_rows[2 * third :],
    )

    write_system_a(rows_a, RAW_DIR / "system_a_shipments.csv")
    write_system_b(rows_b, RAW_DIR / "system_b_shipments.csv")
    write_system_c(rows_c, RAW_DIR / "system_c_shipments.csv")

    print(f"Wrote {len(rows_a)} rows -> {RAW_DIR / 'system_a_shipments.csv'}")
    print(f"Wrote {len(rows_b)} rows -> {RAW_DIR / 'system_b_shipments.csv'}")
    print(f"Wrote {len(rows_c)} rows -> {RAW_DIR / 'system_c_shipments.csv'}")


if __name__ == "__main__":
    main()
