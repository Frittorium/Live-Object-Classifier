# YOLO Object Detection Application

Real-time object detection from a local camera feed using YOLOv8 and OpenCV.

## Quick Start

```bash
pip install ultralytics opencv-python numpy
python src/main.py
```

Press **Q** or close the window to exit.

## Requirements

- Python 3.8+
- ultralytics
- opencv-python
- numpy
- torch (installed automatically with ultralytics)

## Project Structure
```text
yolo-object-detection-app/
├── documents/                  # SAD and SDP documents
├── src/
│   ├── __init__.py
│   ├── main.py                # Application entry point and controller
│   ├── config.py              # Configuration and validation
│   ├── logger.py              # Centralized logging (console + rotating file)
│   ├── camera.py              # Camera abstraction with recovery logic
│   ├── preprocessing.py       # Frame validation and preprocessing
│   ├── detection.py           # Detection data model and validation
│   ├── detector.py            # YOLO model interface (vendor isolation)
│   ├── renderer.py            # Rendering and OpenCV display
│   └── errors.py              # Custom exception types
├── models/                    # Model weights directory
│   └── yolov8n.pt             # Auto-downloaded on first run or place manually
├── logs/                      # Application logs (created at runtime)
├── .gitignore
├── README.md
└── requirements.txt
```


## Usage

### Basic Run

```bash
python src/main.py
```

### Configuration

Edit `src/config.py` to adjust:

- **camera_index**: Camera device (default: `0`)
- **camera_width, camera_height**: Resolution (default: `640x480`)
- **model_path**: Path to YOLO weights (default: `models/yolov8n.pt`)
- **device**: `"cpu"`, `"cuda"`, or `"auto"` (default: `"cpu"`)
- **confidence_threshold**: Detection confidence filter (default: `0.50`)
- **exit_key**: Key to quit (default: `"q"`)

### Logging

Logs are written to `logs/app.log` and rotated automatically. Console output shows INFO and above.

## Model Information

YOLOv8 Nano (`yolov8n.pt`) detects 80 COCO classes:

person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

### Using Larger Models

For better accuracy at the cost of speed:

- `yolov8s.pt` (small)
- `yolov8m.pt` (medium)
- `yolov8l.pt` (large)
- `yolov8x.pt` (extra large)

Edit `config.py`:
```python
model_path: str = "models/yolov8l.pt"
```

Models are auto-downloaded on first use or place the `.pt` file in `models/`.

## Architecture

The application follows a strict separation of concerns:

- **Camera**: Captures frames with bounded recovery
- **Preprocessing**: Validates frames before inference
- **Detector**: Isolated YOLO interface (vendor-specific code contained)
- **Detection Processing**: Validates and filters detections
- **Renderer**: Draws bounding boxes and handles display/input

This design allows components to be tested independently and replaced without cascading changes.

## Performance

Typical performance on reference hardware (CPU-only, YOLOv8n, 640×480):

- ~15–20 FPS on moderate CPU
- ~30+ FPS on GPU (CUDA)
- Inference latency ~30–50 ms per frame (CPU)

Adjust resolution and model size based on your hardware.

## Failure Handling

- **Camera unavailable**: Clean shutdown with logged error
- **Camera read failures**: Bounded retry then reconnect with limits
- **Inference failures**: Skips frame and escalates after threshold
- **Invalid frames**: Silently skipped; persistent invalidity triggers camera recovery
- **Display failure**: Controlled shutdown

All errors are logged to console and file.

## Logging

- **INFO**: Lifecycle events (startup, camera open, model load, shutdown)
- **DEBUG**: Frame details, timing, detection counts (when enabled)
- **WARNING**: Recoverable failures (transient read errors, rate-limited repeats)
- **ERROR**: Failed operations (recovery exhausted, display error)
- **CRITICAL**: Fatal startup failures (missing model, camera unavailable)

Repeated errors are rate-limited to avoid log flooding.

## Known Limitations

- Single-threaded; FPS is bounded by combined latency of all stages
- Overlapping objects may not all be detected (YOLO design limitation; use larger model or lower confidence threshold if needed)
- No object tracking or counting (V1 scope)
- No recording or frame persistence

## Testing

Unit and integration tests can be added in `tests/` using mock camera/detector/renderer. The controller constructor accepts these as parameters for testability.

## License

MIT