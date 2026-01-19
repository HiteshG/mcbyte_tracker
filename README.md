# 🏒 Hockey Multi-Object Tracking with McByte

Production-ready multi-object tracking system for ice hockey video analysis, combining custom YOLO detection with McByte tracking and Cutie mask propagation.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Technologies & Concepts](#technologies--concepts)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Configuration](#configuration)
7. [Integration Guide](#integration-guide)
8. [API Reference](#api-reference)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting](#troubleshooting)

---

## Overview

This system provides end-to-end multi-object tracking for hockey games, designed to work with your custom YOLO detection model trained on hockey-specific classes:

```python
CLASS_NAMES = {
    0: "Center Ice",
    1: "Faceoff",
    2: "Goalpost",
    3: "Goaltender",  # Tracked
    4: "Player",      # Tracked
    5: "Puck",        # Tracked
    6: "Referee"      # Tracked
}
```

### Key Features

- ✅ **Custom YOLO Integration**: Works with any Ultralytics YOLO model (v5, v8, v11, etc.)
- ✅ **McByte Tracking**: State-of-the-art ByteTrack-based tracker with mask propagation
- ✅ **Temporal Mask Propagation**: Uses Cutie + SAM for robust object segmentation
- ✅ **Production Ready**: Modular, documented, and optimized code
- ✅ **Flexible Configuration**: Easily adjustable parameters for different scenarios

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hockey Tracking Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │   Video      │    │    YOLO      │    │     McByte       │  │
│  │   Input      │───▶│  Detection   │───▶│    Tracker       │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│                              │                    │              │
│                              │                    │              │
│                              ▼                    ▼              │
│                      ┌──────────────┐    ┌──────────────────┐  │
│                      │     SAM      │    │    Kalman        │  │
│                      │    (Init)    │    │    Filter        │  │
│                      └──────────────┘    └──────────────────┘  │
│                              │                    │              │
│                              ▼                    │              │
│                      ┌──────────────┐            │              │
│                      │    Cutie     │────────────┘              │
│                      │ (Propagate)  │                           │
│                      └──────────────┘                           │
│                              │                                   │
│                              ▼                                   │
│                      ┌──────────────────────────────────────┐   │
│                      │         Output: Track Results         │   │
│                      │   (IDs, Bboxes, Classes, Masks)       │   │
│                      └──────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
hockey_mcbyte_tracker/
├── configs/
│   ├── __init__.py
│   └── config.py           # Configuration dataclasses
├── core/
│   ├── __init__.py
│   └── pipeline.py         # Main tracking pipeline
├── detectors/
│   ├── __init__.py
│   └── yolo_detector.py    # YOLO detection wrapper
├── tracking/
│   ├── __init__.py
│   ├── kalman_filter.py    # Kalman filter implementation
│   ├── track.py            # Track class and matching
│   └── mcbyte_tracker.py   # McByte tracker
├── mask_propagation/
│   ├── __init__.py
│   └── mask_propagator.py  # Cutie + SAM integration
├── utils/
│   ├── __init__.py
│   └── visualization.py    # Drawing utilities
├── notebooks/
│   └── hockey_tracking.ipynb  # Tutorial notebook
├── weights/                # Model weights directory
├── outputs/                # Output directory
├── __init__.py
└── README.md
```

---

## Technologies & Concepts

### 1. YOLO (You Only Look Once)

**What it is**: A real-time object detection algorithm that processes the entire image in one pass.

**How it's used**: Your custom YOLO model detects hockey objects (players, puck, referee, goaltender) and provides bounding boxes with confidence scores.

**Key concepts**:
- **Confidence threshold**: Minimum detection confidence (0.25 default)
- **NMS (Non-Maximum Suppression)**: Removes overlapping detections
- **Class filtering**: Only track specific classes (players, not goalposts)

```python
# Detection output format
Detection(
    bbox=[x1, y1, x2, y2],  # Bounding box coordinates
    confidence=0.85,         # Detection confidence
    class_id=4,              # Player class
    class_name="Player"
)
```

### 2. ByteTrack / McByte Tracking

**What it is**: A tracking-by-detection paradigm that associates detections across frames using motion prediction and appearance matching.

**How it works**:
1. **High-confidence association**: Match high-score detections to existing tracks
2. **Low-confidence association**: Match remaining tracks with low-score detections
3. **Track lifecycle**: Manage track states (NEW → TRACKED → LOST → REMOVED)

**Key concepts**:
- **IoU (Intersection over Union)**: Measures overlap between predicted and detected boxes
- **Hungarian algorithm**: Optimal bipartite matching for track-detection association
- **Track buffer**: Number of frames to keep lost tracks before deletion

```python
# Association flow
tracks = tracker.update(detections, frame)

# Track states
TrackState.NEW       # Just created, waiting for confirmation
TrackState.TRACKED   # Actively being tracked
TrackState.LOST      # Temporarily lost, waiting to reappear
TrackState.REMOVED   # Permanently deleted
```

### 3. Kalman Filter

**What it is**: A recursive algorithm for predicting object state (position, velocity) over time.

**How it's used**: Predicts where each track will be in the next frame, even if detection fails.

**State vector**: `[x, y, w, h, vx, vy, vw, vh]`
- `(x, y)`: Bounding box center
- `(w, h)`: Width and height
- `(vx, vy, vw, vh)`: Velocities

```python
# Kalman filter cycle
track.predict()   # Predict next position
track.update(det) # Correct with actual detection
```

### 4. Camera Motion Compensation (GMC)

**What it is**: Estimates and compensates for camera movement between frames.

**How it's used**: Adjusts track predictions when the camera pans/zooms, preventing ID switches.

**Methods available**:
- **Sparse Optical Flow**: Fast, uses corner features
- **ORB**: Feature-based matching
- **ECC**: Enhanced Correlation Coefficient

### 5. SAM (Segment Anything Model)

**What it is**: Foundation model for image segmentation that can segment any object given a prompt (bounding box, point).

**How it's used**: Generates initial segmentation masks from detection bounding boxes.

```python
# SAM generates masks from boxes
masks = sam_predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,  # Detection bboxes
    multimask_output=False
)
```

### 6. Cutie (Video Object Segmentation)

**What it is**: Temporal mask propagation network that tracks masks across video frames.

**How it's used**: Propagates SAM-initialized masks through time, providing robust segmentation even when objects overlap.

**Key features**:
- Object-level memory reading
- Long-term memory for extended sequences
- Handles occlusion and reappearance

```python
# Cutie propagation
mask_torch = processor.step(image_torch, initial_mask, idx_mask=False)  # Init
propagated = processor.step(next_frame_torch)  # Propagate
```

### 7. Mask-Enhanced Association

**What it is**: Using segmentation masks as an additional cue for track-detection association.

**How it improves tracking**:
- IoU computed on masks is more accurate than bounding boxes
- Handles partial occlusions better
- Reduces ID switches in crowded scenes

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+ with CUDA support (recommended)
- NVIDIA GPU with 8GB+ VRAM (for mask propagation)

### Step 1: Core Dependencies

```bash
# Create environment
conda create -n hockey_tracker python=3.9
conda activate hockey_tracker

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install ultralytics opencv-python scipy numpy tqdm matplotlib
```

### Step 2: Mask Propagation (Optional)

```bash
# SAM (Segment Anything)
pip install git+https://github.com/facebookresearch/segment-anything.git

# Cutie (Video Object Segmentation)
git clone https://github.com/hkchengrex/Cutie.git
cd Cutie && pip install -e .
```

### Step 3: Download Weights

```bash
mkdir weights

# Download Cutie weights
wget https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth -P weights/

# Download SAM weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P weights/

# Copy your YOLO model
cp /path/to/your/hockey_yolo.pt weights/
```

---

## Quick Start

### Option 1: High-Level Pipeline

```python
from core.pipeline import run_tracking

# Run tracking with minimal setup
results = run_tracking(
    input_path="video.mp4",
    model_path="weights/hockey_yolo.pt",
    output_path="outputs/tracked.mp4",
    track_classes=[3, 4, 5, 6],  # Goaltender, Player, Puck, Referee
    use_masks=True
)
```

### Option 2: Component-by-Component

```python
from configs.config import PipelineConfig
from detectors.yolo_detector import create_detector
from tracking.mcbyte_tracker import McByteTracker
from tracking.track import Track
from utils.visualization import Visualizer
import cv2

# Configuration
config = PipelineConfig()
config.detector.model_path = "weights/hockey_yolo.pt"
config.detector.track_classes = [3, 4, 5, 6]

# Initialize components
detector = create_detector(config.detector)
tracker = McByteTracker(track_thresh=0.5, track_buffer=30)
visualizer = Visualizer(class_names=config.detector.class_names)

# Process video
cap = cv2.VideoCapture("video.mp4")
Track.reset_id()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect
    detections = detector.detect(frame)
    
    # Track
    tracks = tracker.update(detections, frame)
    
    # Visualize
    vis = visualizer.draw_tracks(frame, tracks)
    
    cv2.imshow("Tracking", vis)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Option 3: Jupyter Notebook

Open `notebooks/hockey_tracking.ipynb` for an interactive tutorial.

---

## Configuration

### Detector Configuration

```python
DetectorConfig(
    model_path="weights/hockey_yolo.pt",  # Your YOLO model
    confidence_threshold=0.25,            # Min detection confidence
    nms_threshold=0.45,                   # NMS IoU threshold
    track_classes=[3, 4, 5, 6],           # Classes to track
    device="cuda:0",                      # Processing device
    half_precision=True,                  # FP16 for speed
)
```

### Tracker Configuration

```python
TrackerConfig(
    track_thresh=0.5,       # High confidence threshold
    track_buffer=30,        # Frames to keep lost tracks
    match_thresh=0.8,       # Association IoU threshold
    low_thresh=0.1,         # Low confidence threshold
    new_track_thresh=0.6,   # Min conf for new tracks
    frame_rate=30,          # Video FPS
    use_gmc=True,           # Camera motion compensation
)
```

### Mask Configuration

```python
MaskConfig(
    cutie_weights="weights/cutie-base-mega.pth",
    sam_weights="weights/sam_vit_b_01ec64.pth",
    max_internal_size=480,  # Lower = faster, less accurate
    mem_every=5,            # Memory update frequency
)
```

---

## Integration Guide

### Using Your Custom YOLO Model

1. **Train your model** using Ultralytics YOLO:
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')
   model.train(data='hockey.yaml', epochs=100)
   ```

2. **Export to desired format** (PyTorch recommended):
   ```python
   model.export(format='pytorch')
   ```

3. **Configure the tracker**:
   ```python
   config.detector.model_path = "path/to/your/model.pt"
   config.detector.class_names = {
       0: "Center Ice",
       1: "Faceoff",
       # ... your classes
   }
   config.detector.track_classes = [3, 4, 5, 6]  # IDs to track
   ```

### Custom Detection Integration

If using a non-Ultralytics detector:

```python
from detectors.yolo_detector import FrameDetections, Detection

# Your custom detection function
def my_detector(frame):
    # Your detection logic here
    return [
        Detection(
            bbox=np.array([x1, y1, x2, y2]),
            confidence=conf,
            class_id=cls_id
        )
        for x1, y1, x2, y2, conf, cls_id in your_detections
    ]

# Use with tracker
detections = my_detector(frame)
frame_dets = FrameDetections(detections, frame_id, (h, w))
tracks = tracker.update(frame_dets, frame)
```

---

## API Reference

### HockeyTrackingPipeline

```python
class HockeyTrackingPipeline:
    def __init__(self, config: PipelineConfig = None)
    def process_frame(self, frame: np.ndarray, use_masks: bool = True) -> FrameResult
    def visualize(self, result: FrameResult) -> np.ndarray
    def run(self, input_source: str, output_path: str = None, ...) -> List[FrameResult]
    def save_mot_results(self, results: List[FrameResult], output_path: str)
    def reset(self)
```

### McByteTracker

```python
class McByteTracker:
    def __init__(self, track_thresh, track_buffer, match_thresh, ...)
    def update(self, detections, frame, mask_info=None) -> List[Track]
    def get_track_by_id(self, track_id: int) -> Optional[Track]
    @property
    def active_track_count(self) -> int
```

### Track

```python
class Track:
    track_id: int           # Unique track identifier
    class_id: int           # Object class
    score: float            # Last detection confidence
    tlwh: np.ndarray        # [top, left, width, height]
    tlbr: np.ndarray        # [top, left, bottom, right]
    state: TrackState       # Current track state
    history: deque          # Position history
```

---

## Performance Optimization

### Speed Improvements

1. **Disable mask propagation** for real-time processing:
   ```python
   results = pipeline.run(input_path, use_masks=False)
   ```

2. **Reduce mask resolution**:
   ```python
   config.mask.max_internal_size = 320  # Default: 480
   ```

3. **Use FP16 inference**:
   ```python
   config.detector.half_precision = True
   ```

4. **Skip frames** for non-real-time analysis:
   ```python
   for i, (frame_id, frame) in enumerate(reader):
       if i % 2 == 0:  # Process every other frame
           continue
   ```

### Memory Optimization

1. **Reduce track buffer**:
   ```python
   config.tracker.track_buffer = 15  # Default: 30
   ```

2. **Lower Cutie memory**:
   ```yaml
   # In cutie config
   max_mem_frames: 5  # Default: 10
   ```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `max_internal_size`, disable masks, use FP16 |
| Track IDs jumping | Increase `track_buffer`, lower `match_thresh` |
| Missing detections | Lower `confidence_threshold` in detector |
| Slow processing | Disable masks, use smaller input resolution |
| Import errors | Ensure all packages installed, check Python path |

### Debug Mode

```python
# Enable verbose logging
config.verbose = True

# Save intermediate results
config.save_logs = True

# Run with debug visualization
for result in pipeline.run(input_path, show_preview=True):
    print(f"Frame {result.frame_id}: {result.num_tracks} tracks")
```

---

## Citation

If you use this code in your research:

```bibtex
@InProceedings{Stanczyk_2025_CVPR,
    author    = {Stanczyk, Tomasz and Yoon, Seongro and Bremond, Francois},
    title     = {No Train Yet Gain: Towards Generic Multi-Object Tracking in Sports and Beyond},
    booktitle = {CVPR Workshops},
    year      = {2025}
}

@inproceedings{cheng2023putting,
    title={Putting the Object Back into Video Object Segmentation},
    author={Cheng, Ho Kei and Oh, Seoung Wug and Price, Brian and Lee, Joon-Young and Schwing, Alexander},
    booktitle={CVPR},
    year={2024}
}
```

---

## License

This project combines code from multiple sources:
- McByte tracking: See original repository license
- Cutie: MIT License
- SAM: Apache 2.0 License

---

## Support

For issues and questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the Jupyter notebook for examples
3. Open an issue with reproducible code
