"""
Thin adapter around ultralytics' YOLOv8 + native ByteTrack/BoT-SORT.

This is the ONLY module that imports `ultralytics` or knows the shape of
its Results objects. Everything downstream consumes plain Detection
objects with already-clipped, already-int BoundingBoxes -- per SPEC.md
Section 5 DO ("use model.track(..., persist=True)") and Section 3.C
(clip + int-cast guardrail, enforced here at the source).
"""
from __future__ import annotations

import numpy as np

from .geometry import clip_bbox
from .schemas import Detection

# COCO class indices for target road participants, per SPEC.md Section 3.A
TARGET_CLASSES: dict[int, str] = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class VehicleTracker:
    def __init__(self, model_path: str = "yolov8n.pt") -> None:
        # Imported lazily so pure-logic modules/tests never require torch.
        from ultralytics import YOLO
        self._model = YOLO(model_path)

    def track_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        frame_w: int,
        frame_h: int,
        imgsz: int = 640,
    ) -> list[Detection]:
        """Run tracking on a single frame and return only target-class
        vehicles, each with an already clipped + int-cast BoundingBox.
        """
        results = self._model.track(
            frame,
            persist=True,
            verbose=False,
            imgsz=imgsz,  # Resizes input for significantly faster CPU inference
            classes=list(TARGET_CLASSES.keys()),  # Filters classes during inference
        )
        return self._parse_results(results, frame_index, frame_w, frame_h)

    def _parse_results(
        self,
        results,
        frame_index: int,
        frame_w: int,
        frame_h: int,
    ) -> list[Detection]:
        detections: list[Detection] = []
        if not results:
            return detections

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.id is None:
            # No tracked boxes this frame (e.g. nothing detected, or the
            # tracker hasn't assigned IDs yet) -- not an error.
            return detections

        ids = boxes.id.int().tolist()
        classes = boxes.cls.int().tolist()
        xyxy = boxes.xyxy.tolist()

        for track_id, class_id, (x1, y1, x2, y2) in zip(ids, classes, xyxy):
            if class_id not in TARGET_CLASSES:
                continue
            bbox = clip_bbox(x1, y1, x2, y2, frame_w, frame_h)
            detections.append(
                Detection(
                    vehicle_id=int(track_id),
                    class_id=int(class_id),
                    bbox=bbox,
                    frame_index=frame_index,
                )
            )
        return detections