# Hockey McByte Tracker v2.0

## Occlusion-Robust Multi-Object Tracking for Hockey

A production-ready tracking system specifically designed for hockey's unique challenges: fast motion, frequent occlusions, similar appearances, and identity preservation through collisions.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [The Occlusion Problem in Hockey](#the-occlusion-problem-in-hockey)
3. [Architecture Deep Dive](#architecture-deep-dive)
4. [Core Components](#core-components)
   - [Unscented Kalman Filter](#1-unscented-kalman-filter-ukf)
   - [SAM Mask Generation](#2-sam-mask-generation)
   - [CUTIE Mask Propagation](#3-cutie-mask-propagation)
   - [Appearance-Based Re-Identification](#4-appearance-based-re-identification)
   - [Multi-Cue Association](#5-multi-cue-association)
5. [How Components Work Together](#how-components-work-together)
6. [Identity Preservation Pipeline](#identity-preservation-pipeline)
7. [Camera Motion Compensation](#camera-motion-compensation)
8. [Configuration Guide](#configuration-guide)
9. [Performance Characteristics](#performance-characteristics)
10. [Installation](#installation)
11. [Quick Start](#quick-start)
12. [Troubleshooting](#troubleshooting)

---

## System Overview

This tracker solves a fundamental problem in hockey video analysis: **maintaining consistent player identities through occlusions**. When two players collide, overlap, or move behind each other, traditional trackers lose track and assign new IDs when they reappear. This system preserves identity through a multi-layered approach.

### Key Capabilities

| Capability | Benefit |
|------------|---------|
| Non-linear motion modeling | Handles sudden direction changes, stops, accelerations |
| Pixel-precise segmentation | Knows exactly which pixels belong to which player |
| Temporal mask propagation | Maintains identity even when detection fails |
| Appearance matching | Re-identifies players after long occlusions |
| Camera motion compensation | Works with moving broadcast cameras |

---

## The Occlusion Problem in Hockey

Hockey presents unique tracking challenges that this system addresses:

### Why Traditional Trackers Fail

1. **Players look similar**: Same team jerseys, similar body shapes
2. **Constant contact**: Body checks, scrums near the goal, pile-ups
3. **Fast motion**: Players can cross the frame in under a second
4. **Detection gaps**: Puck and partially occluded players often missed
5. **Camera motion**: Broadcast cameras pan, zoom, and cut

### Our Solution

Instead of relying solely on bounding box IoU (Intersection over Union), we combine:

- **Motion prediction** that understands non-linear trajectories
- **Pixel-level masks** that survive partial occlusions
- **Appearance features** that recognize players by their visual characteristics
- **Track memory** that remembers lost players for several seconds

---

## Architecture Deep Dive

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT: Video Frame                           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     1. CAMERA MOTION COMPENSATION                   │
│                                                                     │
│   Sparse Optical Flow → Affine Transform → Warp Track Predictions  │
│                                                                     │
│   Effect: Track positions remain valid even when camera moves       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      2. YOLO DETECTION                              │
│                                                                     │
│   Custom hockey model detects: Players, Goalies, Puck, Referees    │
│   Output: Bounding boxes with class and confidence                  │
│                                                                     │
│   Effect: Provides candidates for tracking each frame               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    3. UKF STATE PREDICTION                          │
│                                                                     │
│   For each existing track:                                          │
│   - Generate sigma points around current state                      │
│   - Propagate through non-linear motion model                       │
│   - Compute predicted mean and covariance                           │
│                                                                     │
│   Effect: Predicts where each player should be this frame           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    4. MASK PROPAGATION (CUTIE)                      │
│                                                                     │
│   Propagate masks from previous frame to current frame              │
│   Uses temporal memory to maintain segmentation through occlusion   │
│                                                                     │
│   Effect: Know which pixels belong to which track even without      │
│           a new detection                                           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   5. FEATURE EXTRACTION                             │
│                                                                     │
│   For each detection:                                               │
│   - Color histogram (HSV for illumination robustness)               │
│   - Deep CNN features (optional, for stronger ReID)                 │
│   - SAM mask generation (for new tracks)                            │
│                                                                     │
│   Effect: Creates signatures for matching and re-identification     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  6. MULTI-CUE ASSOCIATION                           │
│                                                                     │
│   Build cost matrix combining:                                      │
│   - Motion cost (1 - IoU between predicted and detected boxes)      │
│   - Appearance cost (feature distance)                              │
│   - Mask cost (1 - mask IoU)                                        │
│                                                                     │
│   Solve assignment with Hungarian algorithm                         │
│                                                                     │
│   Effect: Optimal matching considering all available information    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    7. CASCADED MATCHING                             │
│                                                                     │
│   Stage 1: Confirmed tracks ↔ High-confidence detections            │
│   Stage 2: Remaining + Lost tracks ↔ Remaining detections           │
│   Stage 3: Unconfirmed tracks ↔ Remaining detections                │
│   Stage 4: Unmatched tracks ↔ Low-confidence detections             │
│                                                                     │
│   Effect: Prioritizes reliable matches, salvages difficult ones     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    8. STATE UPDATE                                  │
│                                                                     │
│   Matched tracks: UKF update with measurement                       │
│   Unmatched tracks: Mark as lost, store in memory                   │
│   Unmatched detections: Create new tracks                           │
│                                                                     │
│   Effect: Maintains track states and handles lifecycle              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        OUTPUT                                       │
│                                                                     │
│   - Track IDs with bounding boxes                                   │
│   - Segmentation masks per track                                    │
│   - Trajectories and velocities                                     │
│   - Confidence scores                                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Unscented Kalman Filter (UKF)

#### What It Does

The UKF estimates and predicts player positions using a probabilistic state model. Unlike the standard linear Kalman filter, it handles non-linear motion patterns.

#### Why Non-Linear Matters for Hockey

Hockey players don't move in straight lines. They:
- Stop suddenly
- Change direction sharply
- Accelerate and decelerate rapidly
- Move in curves around opponents

A linear Kalman filter assumes constant velocity in a straight line. When a player turns, the linear prediction is wrong, causing tracking failures.

#### How UKF Works

1. **State Representation**: Instead of tracking `[x, y, width, height]`, we use `[x, y, scale, aspect_ratio, vx, vy, v_scale]`. Scale (area) is more stable than width/height independently.

2. **Sigma Points**: The UKF generates a small set of carefully chosen sample points (sigma points) that capture the statistical distribution of the state.

3. **Non-linear Propagation**: These points are passed through the actual motion model (which can be non-linear), then recombined to get the predicted distribution.

4. **Adaptive Noise**: Process noise increases for faster-moving objects, acknowledging greater uncertainty in rapid motion.

#### Effect on Pipeline

- More accurate predictions = better IoU matching
- Handles direction changes without losing tracks
- Provides uncertainty estimates for gating unrealistic matches

---

### 2. SAM Mask Generation

#### What It Does

SAM (Segment Anything Model) generates pixel-precise segmentation masks from bounding boxes. Given a detection box, it produces an exact outline of the player.

#### Why Masks Beat Boxes

Bounding boxes are rectangles that include background. When two players overlap:
- Boxes have high IoU even if players are separate
- Boxes can't distinguish which pixels belong to whom

Masks provide:
- Exact player boundaries
- Ability to detect partial occlusion (mask area shrinks)
- Better appearance features (only player pixels, no background)

#### How SAM Integrates

1. **New Track**: When a track is created, SAM generates its initial mask from the detection box
2. **Track Recovery**: When re-identifying a lost track, SAM provides a mask for comparison
3. **Quality Assessment**: Mask area vs. box area ratio indicates detection quality

#### Effect on Pipeline

- Cleaner appearance features (no background contamination)
- Precise occlusion detection
- Higher-quality track initialization

---

### 3. CUTIE Mask Propagation

#### What It Does

CUTIE propagates segmentation masks through time. Given masks in frame N, it predicts where those masks should be in frame N+1, even without new detections.

#### Why Temporal Propagation Matters

Detection isn't perfect. Players get missed when:
- Partially occluded
- Motion blur
- Unusual poses
- Detector confidence fluctuates

Without propagation, a missed detection = lost track. With CUTIE:
- Masks persist through detection gaps
- Identity maintained by pixel continuity
- Track survives several frames without detection

#### How CUTIE Works

1. **Memory Bank**: Stores feature representations of each tracked object
2. **Attention Mechanism**: Matches stored features to current frame
3. **Mask Prediction**: Outputs segmentation mask for each object
4. **Memory Update**: Periodically updates stored features

#### Effect on Pipeline

- Tracks survive detection dropouts
- Smooth mask boundaries through time
- Identity preserved through brief occlusions
- Provides mask IoU for association even without detection

---

### 4. Appearance-Based Re-Identification

#### What It Does

Extracts visual features that describe what each player looks like, enabling recognition after long occlusions.

#### Feature Types

**Color Histograms**
- Fast to compute
- Captures jersey color, skin tone
- Uses HSV color space for illumination robustness
- Compared using Bhattacharyya distance

**Deep CNN Features** (Optional)
- More discriminative
- Captures texture, patterns, body shape
- Uses models like OSNet trained for person ReID
- Compared using cosine similarity

#### Feature Gallery

Each track maintains a gallery of recent features:
- New features added when quality is high
- Weighted average for smooth representation
- Recent features weighted more heavily
- Survives appearance variations (turning, lighting)

#### Effect on Pipeline

- Re-identify players after several seconds of occlusion
- Distinguish similar-looking players
- Robust to temporary appearance changes
- Enables track recovery without position continuity

---

### 5. Multi-Cue Association

#### What It Does

Combines motion, appearance, and mask information into a single cost matrix for optimal track-detection matching.

#### Cost Components

**Motion Cost** (weight: 0.4)
- Based on IoU between predicted track box and detection box
- High IoU = low cost
- Fast, works well for frame-to-frame matching
- Fails during occlusion

**Appearance Cost** (weight: 0.35)
- Based on feature distance between track gallery and detection
- Similar appearance = low cost
- Works across occlusions
- Can be fooled by teammates

**Mask Cost** (weight: 0.25)
- Based on IoU between propagated mask and detection mask
- Pixel-level agreement = low cost
- Robust to box ambiguity
- Most precise during partial occlusion

#### Fusion Strategy

```
Total Cost = 0.4 × Motion + 0.35 × Appearance + 0.25 × Mask
```

The weights balance:
- Speed (motion is fastest)
- Robustness (appearance survives occlusion)
- Precision (masks are most accurate)

#### Effect on Pipeline

- No single failure mode breaks tracking
- Graceful degradation when components fail
- Optimal assignment considers all evidence
- Adjustable weights for different scenarios

---

## How Components Work Together

### Normal Tracking (No Occlusion)

1. YOLO detects player with high confidence
2. UKF predicts where track should be
3. Motion IoU is high → strong match signal
4. Appearance confirms identity
5. Mask provides pixel precision
6. Track updated, everyone happy

### During Occlusion

1. Two players collide, detection merges or fails
2. UKF predictions diverge (players moving different directions)
3. CUTIE propagates individual masks through occlusion
4. Mask IoU disambiguates which detection belongs to which track
5. Appearance features provide backup identity signal
6. Tracks maintained with correct IDs

### After Occlusion (Re-identification)

1. Player reappears after being lost for 30 frames
2. UKF prediction is stale but provides search region
3. Appearance matching finds similar features
4. Mask verification confirms spatial consistency
5. Track re-activated with original ID
6. Continuity preserved for analytics

---

## Identity Preservation Pipeline

This is the key innovation: a dedicated pipeline for keeping track IDs consistent.

### Track States

| State | Meaning | Duration |
|-------|---------|----------|
| NEW | Just created, awaiting confirmation | 1-3 frames |
| TRACKED | Actively matched to detections | Until lost |
| LOST | No recent match, searching | Up to 2 seconds |
| REMOVED | Too long without match | Permanent |

### Track Memory

When a track enters LOST state:
1. **Position + Velocity** stored for prediction
2. **Appearance Gallery** preserved for matching
3. **Last Mask** saved for verification
4. **Lost Frame** recorded for timeout

### Re-identification Process

1. Detection appears without matching active track
2. System queries lost track memory
3. Position check: Is detection near predicted location?
4. Appearance check: Do features match gallery?
5. Mask check: Does SAM mask match stored mask?
6. If checks pass → Re-activate track with original ID
7. If checks fail → Create new track

### Timeout Policy

- Tracks kept in memory for `track_buffer` frames (default: 60 = 2 seconds)
- After timeout, track ID released
- Memory cleaned to prevent unbounded growth

---

## Camera Motion Compensation

Hockey broadcasts use moving cameras. Without compensation, track predictions become invalid when the camera pans.

### How It Works

1. **Feature Detection**: Find distinctive points in the frame
2. **Optical Flow**: Track these points to next frame
3. **Transform Estimation**: Compute affine transform (translation + rotation + scale)
4. **State Warping**: Apply transform to all track positions

### Methods Available

| Method | Speed | Accuracy | Best For |
|--------|-------|----------|----------|
| Sparse | Fast | Good | Real-time |
| ORB | Medium | Good | General |
| ECC | Slow | Best | Offline |

### Effect on Pipeline

- Track predictions remain valid during camera motion
- Prevents false track breaks during pans
- Essential for broadcast footage

---

## Configuration Guide

### Speed vs. Accuracy Tradeoff

**Fast Mode** (Real-time, ~30 FPS)
- Masks disabled
- ReID disabled
- Short track buffer (15 frames)
- Higher confidence thresholds

**Balanced Mode** (Default, ~15 FPS)
- Masks enabled
- Color histogram ReID
- Medium track buffer (60 frames)
- Standard thresholds

**Accurate Mode** (Offline, ~5 FPS)
- Full mask propagation
- Deep CNN ReID
- Long track buffer (90 frames)
- Lower thresholds (catch more)

### Key Parameters

| Parameter | Effect | Range |
|-----------|--------|-------|
| `track_buffer` | How long to remember lost tracks | 15-120 frames |
| `motion_weight` | Trust in position prediction | 0.2-0.6 |
| `appearance_weight` | Trust in visual features | 0.2-0.5 |
| `mask_weight` | Trust in pixel masks | 0.1-0.4 |
| `track_high_thresh` | Minimum confidence for primary matching | 0.4-0.6 |
| `track_low_thresh` | Minimum confidence for secondary matching | 0.1-0.2 |

---

## Performance Characteristics

### Expected Performance

| Mode | GPU | FPS | ID Switches (lower=better) |
|------|-----|-----|---------------------------|
| Fast | RTX 3080 | 30+ | Moderate |
| Balanced | RTX 3080 | 15-20 | Low |
| Accurate | RTX 3080 | 5-10 | Very Low |

### Bottlenecks

1. **Detection**: YOLO inference (use half precision)
2. **Mask Propagation**: CUTIE memory attention (limit history)
3. **Feature Extraction**: CNN forward pass (batch if possible)

### Memory Usage

- Base tracker: ~500MB GPU
- With SAM: +1-2GB GPU
- With CUTIE: +1-2GB GPU
- Per-track overhead: ~1MB (features + masks)

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- 8GB+ GPU memory (for full features)

### Basic Installation

```bash
pip install -r requirements.txt
```

### Optional Components

**SAM (Segment Anything)**
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
# Download weights: sam_vit_b_01ec64.pth
```

**CUTIE (Mask Propagation)**
```bash
pip install git+https://github.com/hkchengrex/Cutie.git
# Download weights: cutie-base-mega.pth
```

**BoxMOT (Alternative with built-in ReID)**
```bash
pip install boxmot
```

### Weight Files

Place in `weights/` directory:
- `hockey_yolo.pt` - Your trained YOLO model
- `sam_vit_b_01ec64.pth` - SAM weights (optional)
- `cutie-base-mega.pth` - CUTIE weights (optional)

---

## Quick Start

### Minimal Example

```python
from hockey_mcbyte_tracker import run_tracking

results = run_tracking(
    input_path="hockey_game.mp4",
    output_path="tracked.mp4",
    model_path="weights/hockey_yolo.pt"
)
```

### Full Configuration

```python
from hockey_mcbyte_tracker import HockeyPipeline, PipelineConfig

config = PipelineConfig()
config.detector.model_path = "weights/hockey_yolo.pt"
config.detector.track_classes = [3, 4, 5, 6]  # Goalie, Player, Puck, Ref
config.tracker.track_buffer = 60
config.mask.enabled = True

pipeline = HockeyPipeline(config)
results = pipeline.process_video("input.mp4", "output.mp4")
```

### Frame-by-Frame

```python
from hockey_mcbyte_tracker import OcclusionRobustTracker, TrackerConfig

tracker = OcclusionRobustTracker(TrackerConfig())

for frame in video:
    detections = your_detector(frame)
    result = tracker.update(detections, frame)
    
    for track in result.tracks:
        print(f"Track {track.track_id}: {track.tlbr}")
```

---

## Troubleshooting

### Track IDs Still Switching

**Symptoms**: IDs change after collisions despite full system

**Solutions**:
1. Increase `track_buffer` (keep tracks longer)
2. Lower `appearance_thresh` (accept more ReID matches)
3. Ensure SAM/CUTIE weights are loaded
4. Check GPU memory (components may fail silently)

### Low FPS

**Symptoms**: Processing slower than expected

**Solutions**:
1. Disable masks (`config.mask.enabled = False`)
2. Enable half precision (`config.detector.half_precision = True`)
3. Reduce input resolution
4. Use "Fast" configuration preset

### Tracks Not Created

**Symptoms**: Detections appear but no tracks form

**Solutions**:
1. Lower `new_track_thresh`
2. Check `track_classes` includes your class IDs
3. Verify detector confidence is above `track_low_thresh`

### Memory Growing Unbounded

**Symptoms**: GPU memory increases over time

**Solutions**:
1. Reduce `track_buffer`
2. Call `tracker.reset()` periodically
3. Limit CUTIE memory frames

---

## File Structure

```
hockey_mcbyte_tracker/
├── __init__.py              # Main exports
├── requirements.txt         # Dependencies
├── setup.py                 # Package setup
├── usage_examples.py        # Example code
│
├── core/
│   ├── hockey_pipeline.py   # Main pipeline
│   └── pipeline.py          # Legacy pipeline
│
├── tracking/
│   ├── occlusion_robust_tracker.py  # Main tracker
│   ├── ukf.py               # Unscented Kalman Filter
│   ├── enhanced_track.py    # Track class with ReID
│   ├── appearance.py        # Appearance features
│   ├── kalman_filter.py     # Standard KF (legacy)
│   └── mcbyte_tracker.py    # Original tracker (legacy)
│
├── mask_propagation/
│   ├── integrated_masks.py  # SAM + CUTIE system
│   └── mask_propagator.py   # Legacy masks
│
├── detectors/
│   └── yolo_detector.py     # YOLO integration
│
├── utils/
│   └── visualization.py     # Drawing utilities
│
├── configs/
│   └── config.py            # Configuration classes
│
└── weights/                 # Model weights (not included)
```

---

## Technical Specifications

### State Vector (UKF)

```
State: [x, y, s, r, vx, vy, vs]

x, y    : Bounding box center position
s       : Scale (area = width × height)
r       : Aspect ratio (width / height)
vx, vy  : Velocity components
vs      : Scale velocity
```

### Feature Vector (Appearance)

```
Histogram: 32-dimensional (16 H-bins + 16 S-bins in HSV)
Deep:      512-dimensional (OSNet or similar CNN)
Combined:  Weighted fusion based on availability
```

### Association Cost

```
Cost = w_m × (1 - IoU_box) + w_a × dist_appearance + w_k × (1 - IoU_mask)

Default weights: w_m = 0.4, w_a = 0.35, w_k = 0.25
```

---

## License

MIT License - Use freely for research and commercial applications.

---

## Citation

If you use this tracker in research:

```
@software{hockey_mcbyte_tracker,
  title={Hockey McByte Tracker: Occlusion-Robust Multi-Object Tracking},
  year={2024},
  description={Production tracking with UKF, SAM, and CUTIE integration}
}
```

---

## Acknowledgments

This system builds on:
- **Segment Anything (SAM)** - Meta AI
- **CUTIE** - hkchengrex
- **YOLOv8** - Ultralytics
- **ByteTrack** - Original cascaded matching concept
- **OSNet** - Person re-identification

---

*Built for hockey. Works for any sport with frequent occlusions.*
