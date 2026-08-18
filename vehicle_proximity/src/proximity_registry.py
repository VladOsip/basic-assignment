"""
The only mutable, cross-frame state in the whole pipeline.

Wraps a dict[vehicle_id -> PeakRecord] behind an interface so the
strict-inequality "first frame wins on ties" rule from SPEC.md
Section 3.D can't be accidentally violated (e.g. by a stray `>=`)
anywhere else in the codebase.
"""
from __future__ import annotations

from .schemas import BoundingBox, Detection, PeakRecord, ProximityEvent


class ProximityRegistry:
    def __init__(self) -> None:
        self._records: dict[int, PeakRecord] = {}

    def update(self, detection: Detection, score: float,
               timestamp: float) -> ProximityEvent:
        """Feed one Detection + its computed score/timestamp through the
        registry. Updates the stored peak ONLY if score strictly exceeds
        the current max_score (or no record exists yet for this
        vehicle_id). Strict inequality guarantees that if a vehicle halts
        at peak proximity across multiple frames with an identical score,
        the FIRST frame is retained, per SPEC 3.D.
        """
        vehicle_id = detection.vehicle_id
        existing = self._records.get(vehicle_id)

        is_new_peak = existing is None or score > existing.max_score

        if is_new_peak:
            self._records[vehicle_id] = PeakRecord(
                vehicle_id=vehicle_id,
                max_score=score,
                closest_frame=detection.frame_index,
                timestamp=timestamp,
                bounding_box=detection.bbox,
            )

        return ProximityEvent(
            vehicle_id=vehicle_id,
            frame_index=detection.frame_index,
            score=score,
            bbox=detection.bbox,
            is_new_peak=is_new_peak,
        )

    def all_records(self) -> list[PeakRecord]:
        """Snapshot of every vehicle's peak record, order not guaranteed
        beyond Python's dict insertion order (first-seen vehicle_id first).
        """
        return list(self._records.values())

    def get(self, vehicle_id: int) -> PeakRecord | None:
        return self._records.get(vehicle_id)
