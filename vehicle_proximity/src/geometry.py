"""
Pure math for the proximity pipeline. No I/O, no OpenCV objects, no state.

Every function here is independently unit-testable with plain numbers.
This module is the single place SPEC.md Section 3.C (clipping + int
casting) and Section 3.B (the proximity formula) are implemented.
"""
from __future__ import annotations

from .schemas import BoundingBox

# Default metric weights per SPEC.md Section 3.C
DEFAULT_W_AREA = 0.7
DEFAULT_W_BASE = 0.3


def clip_bbox(x1: float, y1: float, x2: float, y2: float,
              frame_w: int, frame_h: int) -> BoundingBox:
    """Clip a raw (possibly float, possibly out-of-bounds) box to
    [0, 0, frame_w, frame_h] and cast to int, per SPEC 3.C.
    """
    cx1 = min(max(int(round(x1)), 0), frame_w)
    cy1 = min(max(int(round(y1)), 0), frame_h)
    cx2 = min(max(int(round(x2)), 0), frame_w)
    cy2 = min(max(int(round(y2)), 0), frame_h)
    return BoundingBox(x1=cx1, y1=cy1, x2=cx2, y2=cy2)


def compute_area(bbox: BoundingBox) -> int:
    """Area = w * h."""
    return bbox.width * bbox.height


def normalize_area(area: int, frame_w: int, frame_h: int) -> float:
    """Area normalized relative to total frame area, in [0, 1]."""
    total = frame_w * frame_h
    if total <= 0:
        return 0.0
    return area / total


def normalize_base_position(y2: int, frame_h: int) -> float:
    """Base position (bottom edge of the box, y2) normalized relative to
    frame height, in [0, 1]. Higher y2 (closer to the bottom of the
    frame) means the vehicle is lower in the perspective view, i.e.
    closer to the camera.
    """
    if frame_h <= 0:
        return 0.0
    return min(max(y2 / frame_h, 0.0), 1.0)


def proximity_score(bbox: BoundingBox, frame_w: int, frame_h: int,
                     w_area: float = DEFAULT_W_AREA,
                     w_base: float = DEFAULT_W_BASE) -> float:
    """P = (Normalized Area * w_area) + (Normalized Base Position * w_base)."""
    area = compute_area(bbox)
    norm_area = normalize_area(area, frame_w, frame_h)
    norm_base = normalize_base_position(bbox.y2, frame_h)
    return (norm_area * w_area) + (norm_base * w_base)


def frame_to_timestamp(frame_index: int, fps: float) -> float:
    """timestamp = round(frame_index / FPS, 2), per SPEC Section 5 DO."""
    if fps <= 0:
        return 0.0
    return round(frame_index / fps, 2)
