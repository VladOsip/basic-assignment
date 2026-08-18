"""
CLI entry point and pipeline orchestration for the Vehicle Proximity
Analysis system (see SPEC.md).

Two-pass design, but with only ONE YOLO inference pass:

  Pass 1 (inference + analysis): read each frame once, run the tracker,
    compute scores, update the ProximityRegistry, and cache the resulting
    Detection list per frame_index (cheap: ints/floats, not pixels).

  Pass 2 (annotation only, no inference): re-read raw frames from disk,
    look up that frame's cached detections, and check whether this frame
    is the now-known GLOBAL peak frame for any vehicle. If so, color it
    red and draw the peak banner. No second call to the model.

This guarantees the highlighted frame is provably the true global peak
per vehicle (unlike a single-pass "provisional flash on every new
personal best"), while avoiding the cost of running inference twice.
Frame *pixels* are still never buffered -- only the lightweight
Detection metadata is cached between passes.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2

from src.annotator import PeakFlashScheduler, draw_detection, draw_peak_banner
from src.exporter import build_json_payload, generate_filename, write_json
from src.geometry import frame_to_timestamp, proximity_score
from src.proximity_registry import ProximityRegistry
from src.schemas import Detection
from src.tracker import VehicleTracker


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect, track, and analyze vehicle proximity in a video."
    )
    parser.add_argument("--input", required=True, help="Path to the input video file.")
    parser.add_argument("--output-dir", default="output",
                         help="Directory for the annotated video + JSON (default: output/).")
    parser.add_argument("--model", default="yolov8n.pt",
                         help="Path or name of the YOLOv8 model weights (default: yolov8n.pt).")
    parser.add_argument("--imgsz", type=int, default=640,
                         help="Inference image size for YOLOv8 model (default: 640).")
    parser.add_argument("--w-area", type=float, default=0.7,
                         help="Weight for normalized area in the proximity score (default: 0.7).")
    parser.add_argument("--w-base", type=float, default=0.3,
                         help="Weight for normalized base position in the proximity score (default: 0.3).")
    return parser.parse_args(argv)


def run_pipeline(input_path: str, output_dir: str, model_path: str,
                  imgsz: int = 640, w_area: float = 0.7, w_base: float = 0.3) -> Path:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # PASS 1: single inference pass -- detect, score, register, cache
    # -------------------------------------------------------------------
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Pass 1/2: Analyzing '{input_path}' ({total_frames} frames)...")

    tracker = VehicleTracker(model_path)
    registry = ProximityRegistry()

    # Cache only lightweight Detection metadata per frame -- NOT pixels.
    # This is what lets Pass 2 skip inference entirely.
    frame_detections: dict[int, list[Detection]] = {}

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if total_frames > 0 and frame_index % 30 == 0:
            pct = (frame_index / total_frames) * 100
            print(f"  Analyzing frame {frame_index}/{total_frames} ({pct:.1f}%)...")

        detections = tracker.track_frame(frame, frame_index, frame_w, frame_h, imgsz=imgsz)
        frame_detections[frame_index] = detections

        timestamp = frame_to_timestamp(frame_index, fps)
        for detection in detections:
            score = proximity_score(detection.bbox, frame_w, frame_h,
                                     w_area=w_area, w_base=w_base)
            registry.update(detection, score, timestamp)

        frame_index += 1

    cap.release()

    # Now that the full video has been seen, each vehicle's TRUE global
    # peak frame is known.
    peak_map: dict[int, int] = {
        rec.vehicle_id: rec.closest_frame for rec in registry.all_records()
    }

    # -------------------------------------------------------------------
    # PASS 2: annotation only -- re-read raw frames, NO second inference
    # -------------------------------------------------------------------
    print("Pass 2/2: Annotating video with global peak events (no re-inference)...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not re-open video file: {input_path}")

    video_filename = f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    video_path = output_dir_path / video_filename
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (frame_w, frame_h))

    flash_scheduler = PeakFlashScheduler()

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if total_frames > 0 and frame_index % 30 == 0:
                pct = (frame_index / total_frames) * 100
                print(f"  Annotating frame {frame_index}/{total_frames} ({pct:.1f}%)...")

            detections = frame_detections.get(frame_index, [])

            # A detection's frame is the global peak iff it matches the
            # value already established in Pass 1 -- no ambiguity, no
            # provisional/repeated flashing.
            for detection in detections:
                if peak_map.get(detection.vehicle_id) == frame_index:
                    flash_scheduler.register_peak(detection.vehicle_id, frame_index)

            active_flashes = flash_scheduler.active_flashes(frame_index)
            for detection in detections:
                flashing = detection.vehicle_id in active_flashes
                draw_detection(frame, detection, flashing=flashing)
            for vehicle_id in active_flashes:
                draw_peak_banner(frame, vehicle_id)

            writer.write(frame)
            frame_index += 1

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    payload = build_json_payload(registry.all_records(), generated_at=datetime.now())
    json_path = output_dir_path / generate_filename()
    write_json(payload, json_path)

    print(f"Done! Annotated video: {video_path}")
    print(f"Done! JSON summary: {json_path}")
    return json_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_pipeline(
            input_path=args.input,
            output_dir=args.output_dir,
            model_path=args.model,
            imgsz=args.imgsz,
            w_area=args.w_area,
            w_base=args.w_base,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
