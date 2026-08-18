# Vehicle Proximity Analysis

A lightweight Python system that detects, tracks, and analyzes vehicles in a
recorded video. For each unique vehicle it identifies the exact frame where
that vehicle reaches peak visual proximity to the camera, then produces:

- A structured **JSON summary** of every vehicle's peak proximity event.
- An **annotated video** with live tracking boxes and a red highlight/banner
  on the frame recognized as each vehicle's closest approach.

Built against `SPEC.md` (Vehicle Proximity Analysis) — see that document for
the full functional specification this implementation follows.

---

## How it works

The pipeline runs in **two passes over the video, but only one YOLO
inference pass**:

1. **Pass 1 (analysis):** reads the video frame by frame, runs YOLOv8 +
   ByteTrack once per frame, computes each detection's proximity score, and
   updates a running "best score so far" record per vehicle. The lightweight
   detection metadata (not pixels) is cached per frame so it doesn't need to
   be recomputed.
2. **Pass 2 (annotation):** once the true global peak frame for every
   vehicle is known, the video is re-read and each frame is annotated —
   drawing tracking boxes in green, and switching to red with a
   "PEAK PROXIMITY" banner for 5 consecutive frames starting at each
   vehicle's actual peak frame. No second inference call is made.

This guarantees the frame marked as "closest" is the true global peak for
that vehicle (not just a provisional local best), without paying the cost of
running the model twice.

### Proximity score

For each bounding box `[x1, y1, x2, y2]`:

```
Area = (x2 - x1) * (y2 - y1)
Score = (normalized_area * 0.7) + (normalized_base_position * 0.3)
```

where `normalized_area` is the box area relative to total frame area, and
`normalized_base_position` is the box's bottom edge (`y2`) relative to frame
height. Larger, lower-in-frame boxes score higher (closer to the camera).

A **strict `>` comparison** is used when updating a vehicle's peak, so if a
vehicle stops and holds the same peak score across several frames, the
*first* frame at that score is recorded — not the last.

---

## Project structure

```
vehicle_proximity/
├── main.py                     # CLI entry point + two-pass pipeline orchestration
├── requirements.txt
├── src/
│   ├── schemas.py               # BoundingBox, Detection, PeakRecord, ProximityEvent
│   ├── geometry.py               # Clipping, area, normalization, scoring, timestamps
│   ├── tracker.py                # YOLOv8 + ByteTrack wrapper (target-class filtering)
│   ├── proximity_registry.py     # Per-vehicle peak-tracking state
│   ├── annotator.py              # Box/label drawing + 5-frame peak flash scheduler
│   └── exporter.py               # JSON schema building + file writing
└── output/                       # Created at runtime: annotated video + JSON summary
```

---

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

On Linux systems with an externally-managed Python environment:

```bash
pip install -r requirements.txt --break-system-packages
```

The YOLOv8 Nano weights (`yolov8n.pt`, ~6MB) are downloaded automatically on
first run if not already present in the working directory.

---

## Usage

```bash
python3 main.py --input car_video_test.mp4 --output-dir output
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--input` | *(required)* | Path to the input video file. |
| `--output-dir` | `output` | Directory for the annotated video + JSON summary. |
| `--model` | `yolov8n.pt` | Path or name of the YOLOv8 model weights. |
| `--imgsz` | `640` | Inference image size. Lower values (e.g. `320`) run faster on CPU at some cost to small-object detection accuracy. |
| `--w-area` | `0.7` | Weight for normalized box area in the proximity score. |
| `--w-base` | `0.3` | Weight for normalized base position (y2) in the proximity score. |

### Example

```bash
python3 main.py --input /path/to/car_video_test.mp4 --output-dir output --imgsz 320
```

---

## Output

Each run produces two files in `--output-dir`:

**`annotated_<timestamp>.mp4`** — the source video re-encoded with:
- Green boxes + `ID: <n>` labels on every tracked vehicle.
- Red boxes and an on-screen `PEAK PROXIMITY - ID: X` banner for 5
  consecutive frames starting at each vehicle's true peak proximity frame.

**`test_YYYYMMDD_HHMMSS.json`** — structured summary:

```json
{
  "generated_at": "2026-08-17T13:15:00",
  "vehicles": [
    {
      "vehicle_id": 1,
      "closest_frame": 142,
      "timestamp": 4.73,
      "bounding_box": [320, 450, 510, 680]
    }
  ]
}
```

- `bounding_box` values are clipped to frame bounds and cast to plain
  integers.
- `timestamp` is `frame_index / fps`, rounded to 2 decimal places.
- An empty `vehicles` list is a valid result (e.g. no target-class objects
  detected in the video) rather than an error.

---

## Target detection classes

Filtered to COCO indices: `bicycle` (1), `car` (2), `motorcycle` (3),
`bus` (5), `truck` (7). All other detected object classes are ignored at
inference time.

---

## Known limitations

- Vehicles that leave and re-enter the frame are treated as new, unrelated
  tracks — there is no re-identification across occlusion/exit gaps, by
  design (per spec).
- Runs a full second pass over the video for annotation, so total wall-clock
  time is roughly one inference pass plus one lightweight I/O/draw pass —
  not strictly single-pass end to end, though inference (the expensive part)
  only happens once.
- Detection/tracking quality depends entirely on YOLOv8 Nano's accuracy;
  small, occluded, or low-contrast vehicles may be missed or lose their
  tracking ID.
