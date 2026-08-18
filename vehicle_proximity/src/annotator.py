"""
Frame-level rendering. Never mutates tracking state -- reads a
ProximityEvent and draws pixels, nothing more.

Owns the "5 consecutive frames" peak-flash timing rule from SPEC.md
Section 4.B, kept in its own scheduler so that rule is independently
testable without a real video frame.
"""
from __future__ import annotations

import cv2
import numpy as np

from .schemas import Detection

GREEN = (0, 255, 0)
RED = (0, 0, 255)
FLASH_DURATION_FRAMES = 5  # per SPEC 4.B


class PeakFlashScheduler:
    """Tracks, per vehicle_id, how many more frames its red "peak" flash
    should render for. A peak registered at frame N flashes for frames
    N, N+1, N+2, N+3, N+4 (5 consecutive frames total) and is inactive
    from N+5 onward.
    """

    def __init__(self) -> None:
        # vehicle_id -> (window_start_frame, window_end_frame_exclusive)
        self._flash_windows: dict[int, tuple[int, int]] = {}

    def register_peak(self, vehicle_id: int, frame_index: int) -> None:
        self._flash_windows[vehicle_id] = (
            frame_index, frame_index + FLASH_DURATION_FRAMES
        )

    def active_flashes(self, frame_index: int) -> set[int]:
        return {
            vid for vid, (start, end) in self._flash_windows.items()
            if start <= frame_index < end
        }


def draw_detection(frame: np.ndarray, detection: Detection,
                    flashing: bool) -> None:
    """Draws the bounding box + 'ID: <n>' label. Green normally, red
    while `flashing` is True (in-place mutation of `frame`, matching the
    frame-by-frame discard-nothing-buffered constraint in SPEC Section 5).
    """
    bbox = detection.bbox
    color = RED if flashing else GREEN
    cv2.rectangle(frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 2)

    label = f"ID: {detection.vehicle_id}"
    label_y = max(bbox.y1 - 8, 12)  # keep label on-screen near the top edge
    cv2.putText(frame, label, (bbox.x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_peak_banner(frame: np.ndarray, vehicle_id: int) -> None:
    """On-screen banner: 'PEAK PROXIMITY - ID: X', per SPEC 4.B."""
    text = f"PEAK PROXIMITY - ID: {vehicle_id}"
    h, w = frame.shape[:2]
    org = (max(w // 2 - 180, 10), 40)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2,
                cv2.LINE_AA)
