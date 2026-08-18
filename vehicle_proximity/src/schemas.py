"""
Data contracts shared across the vehicle proximity pipeline.

These are the ONLY shapes that should cross a module boundary. Every
producer/consumer pair in the pipeline (tracker -> geometry -> registry
-> annotator -> exporter) communicates exclusively through these types.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    """An axis-aligned box already clipped to frame bounds and int-cast.

    Invariant: by the time a BoundingBox exists anywhere in this codebase,
    it has already been through geometry.clip_bbox(). No other module is
    permitted to construct one from raw/unclipped coordinates.
    """
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def as_list(self) -> list[int]:
        """JSON-ready [x_min, y_min, x_max, y_max], plain ints."""
        return [int(self.x1), int(self.y1), int(self.x2), int(self.y2)]


@dataclass
class Detection:
    """A single tracked-object observation in a single frame."""
    vehicle_id: int
    class_id: int
    bbox: BoundingBox
    frame_index: int


@dataclass
class PeakRecord:
    """The best-so-far proximity observation for one vehicle_id."""
    vehicle_id: int
    max_score: float
    closest_frame: int
    timestamp: float
    bounding_box: BoundingBox


@dataclass
class ProximityEvent:
    """Result of feeding one Detection through the registry for one frame."""
    vehicle_id: int
    frame_index: int
    score: float
    bbox: BoundingBox
    is_new_peak: bool
