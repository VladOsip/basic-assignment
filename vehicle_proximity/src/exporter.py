"""
Converts registry state into the exact JSON schema from SPEC.md Section 4.A
and writes it to disk with the required filename format.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .schemas import PeakRecord


def generate_filename(now: datetime | None = None) -> str:
    """test_YYYYMMDD_HHMMSS.json, per SPEC 4.A."""
    now = now or datetime.now()
    return f"test_{now.strftime('%Y%m%d_%H%M%S')}.json"


def build_json_payload(records: list[PeakRecord],
                        generated_at: datetime | None = None) -> dict:
    """Builds the exact schema:
    {
      "generated_at": "<iso timestamp>",
      "vehicles": [
        {"vehicle_id": .., "closest_frame": .., "timestamp": .., "bounding_box": [..]}
      ]
    }
    """
    generated_at = generated_at or datetime.now()
    return {
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "vehicles": [
            {
                "vehicle_id": record.vehicle_id,
                "closest_frame": record.closest_frame,
                "timestamp": round(record.timestamp, 2),
                "bounding_box": record.bounding_box.as_list(),
            }
            for record in records
        ],
    }


def write_json(payload: dict, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)
    return output_path
