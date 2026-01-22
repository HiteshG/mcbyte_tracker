#!/usr/bin/env python3
"""
Hockey Tracker Usage Examples
=============================
Demonstrates all features of the occlusion-robust hockey tracker.

Run: python usage_examples.py
"""

import numpy as np
import cv2
import time
from pathlib import Path


def example_1_quick_start():
    """
    Example 1: Quick Start - Minimal code tracking
    """
    print("\n" + "="*60)
    print("Example 1: Quick Start")
    print("="*60)
    
    from hockey_mcbyte_tracker import run_tracking
    
    # One-liner tracking
    results = run_tracking(
        input_path="hockey_game.mp4",
        output_path="hockey_tracked.mp4",
        model_path="weights/hockey_yolo.pt",
        track_classes=[3, 4, 5, 6],  # Goalie, Player, Puck, Ref
        use_masks=True
    )
    
    print(f"Processed {len(results)} frames")
    for i, result in enumerate(results[:5]):
        print(f"  Frame {i}: {len(result.tracks)} tracks")


def example_2_full_configuration():
    """
    Example 2: Full Configuration - All options
    """
    print("\n" + "="*60)
    print("Example 2: Full Configuration")
    print("="*60)
    
    from hockey_mcbyte_tracker import (
        HockeyPipeline, PipelineConfig, 
        DetectorConfig, TrackerConfig, MaskConfig, VisualizationConfig
    )
    
    # Create configuration
    config = PipelineConfig()
    
    # Detector settings
    config.detector.model_path = "weights/hockey_yolo.pt"
    config.detector.confidence_threshold = 0.3
    config.detector.nms_threshold = 0.45
    config.detector.device = "cuda"
    config.detector.half_precision = True
    config.detector.track_classes = [3, 4, 5, 6]
    config.detector.class_names = {
        0: "Center Ice",
        1: "Faceoff",
        2: "Goalpost",
        3: "Goaltender",
        4: "Player",
        5: "Puck",
        6: "Referee"
    }
    
    # Tracker settings
    config.tracker.track_high_thresh = 0.5
    config.tracker.track_low_thresh = 0.1
    config.tracker.new_track_thresh = 0.6
    config.tracker.match_thresh = 0.8
    config.tracker.track_buffer = 60  # Keep lost tracks for 2 seconds
    config.tracker.motion_weight = 0.4
    config.tracker.appearance_weight = 0.35
    config.tracker.mask_weight = 0.25
    config.tracker.use_gmc = True
    config.tracker.use_appearance = True
    config.tracker.use_masks = True
    config.tracker.use_reid = True
    
    # Mask settings
    config.mask.enabled = True
    config.mask.sam_checkpoint = "weights/sam_vit_b_01ec64.pth"
    config.mask.cutie_checkpoint = "weights/cutie-base-mega.pth"
    
    # Visualization settings
    config.visualization.show_boxes = True
    config.visualization.show_ids = True
    config.visualization.show_masks = True
    config.visualization.show_trajectories = True
    config.visualization.trajectory_length = 30
    
    # Create and run pipeline
    pipeline = HockeyPipeline(config)
    results = pipeline.process_video(
        "hockey_game.mp4",
        "hockey_tracked_full.mp4",
        show_progress=True
    )
    
    # Save MOT format results
    pipeline.save_mot_results("hockey_results.txt")
    
    print(f"Average FPS: {1.0 / np.mean(pipeline.frame_times):.1f}")


def example_3_frame_by_frame():
    """
    Example 3: Frame-by-Frame Processing
    """
    print("\n" + "="*60)
    print("Example 3: Frame-by-Frame Processing")
    print("="*60)
    
    from hockey_mcbyte_tracker import (
        OcclusionRobustTracker, TrackerConfig,
        UltralyticsYOLODetector,
        Visualizer, HOCKEY_CLASS_COLORS
    )
    
    # Initialize components
    detector = UltralyticsYOLODetector(
        model_path="weights/hockey_yolo.pt",
        confidence_threshold=0.3,
        device="cuda"
    )
    
    tracker_config = TrackerConfig()
    tracker_config.use_masks = True
    tracker_config.track_buffer = 60
    tracker = OcclusionRobustTracker(tracker_config)
    
    visualizer = Visualizer(
        class_colors=HOCKEY_CLASS_COLORS,
        show_masks=True,
        show_trajectories=True
    )
    
    # Open video
    cap = cv2.VideoCapture("hockey_game.mp4")
    
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect
        detections = detector.detect(frame)
        
        # Convert to tracker format
        det_list = []
        for det in detections.detections:
            if det.class_id in [3, 4, 5, 6]:  # Hockey classes
                tlwh = np.array([
                    det.bbox[0], det.bbox[1],
                    det.bbox[2] - det.bbox[0],
                    det.bbox[3] - det.bbox[1]
                ])
                det_list.append((tlwh, det.confidence, det.class_id))
        
        # Track
        result = tracker.update(det_list, frame)
        
        # Visualize
        vis_frame = visualizer.draw_tracks(
            frame, result.tracks,
            mask=result.combined_mask,
            track_to_mask={t.track_id: t.track_id for t in result.tracks}
        )
        
        # Display
        cv2.imshow("Tracking", vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Process tracks
        for track in result.tracks:
            print(f"Frame {frame_id}: Track {track.track_id} "
                  f"(class={track.class_id}, score={track.score:.2f}) "
                  f"at {track.tlbr}")
        
        frame_id += 1
    
    cap.release()
    cv2.destroyAllWindows()


def example_4_extract_analytics():
    """
    Example 4: Extract Hockey Analytics
    """
    print("\n" + "="*60)
    print("Example 4: Extract Hockey Analytics")
    print("="*60)
    
    from hockey_mcbyte_tracker import HockeyPipeline, PipelineConfig
    from collections import defaultdict
    
    # Setup
    config = PipelineConfig()
    config.detector.model_path = "weights/hockey_yolo.pt"
    config.detector.track_classes = [3, 4, 5, 6]
    
    pipeline = HockeyPipeline(config)
    
    # Track analytics
    track_positions = defaultdict(list)
    track_classes = {}
    puck_positions = []
    
    # Process video
    cap = cv2.VideoCapture("hockey_game.mp4")
    frame_id = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result, vis_frame = pipeline.process_frame(frame, frame_id)
        
        for track in result.tracks:
            track_positions[track.track_id].append({
                'frame': frame_id,
                'position': track.xywh[:2].tolist(),
                'bbox': track.tlbr.tolist()
            })
            track_classes[track.track_id] = track.class_id
            
            # Track puck separately
            if track.class_id == 5:  # Puck
                puck_positions.append({
                    'frame': frame_id,
                    'position': track.xywh[:2].tolist()
                })
        
        frame_id += 1
    
    cap.release()
    
    # Compute analytics
    print("\n--- Hockey Analytics ---")
    
    # Player speeds
    for track_id, positions in track_positions.items():
        if track_classes.get(track_id) == 4:  # Player
            if len(positions) > 10:
                speeds = []
                for i in range(1, len(positions)):
                    p1 = np.array(positions[i-1]['position'])
                    p2 = np.array(positions[i]['position'])
                    dist = np.linalg.norm(p2 - p1)
                    speeds.append(dist)
                
                avg_speed = np.mean(speeds)
                max_speed = np.max(speeds)
                print(f"Player {track_id}: avg_speed={avg_speed:.1f}, max_speed={max_speed:.1f}")
    
    # Puck statistics
    if puck_positions:
        print(f"\nPuck tracked for {len(puck_positions)} frames")


def example_5_without_masks():
    """
    Example 5: Fast Mode (No Masks)
    For real-time processing
    """
    print("\n" + "="*60)
    print("Example 5: Fast Mode (No Masks)")
    print("="*60)
    
    from hockey_mcbyte_tracker import HockeyPipeline, PipelineConfig
    
    config = PipelineConfig()
    config.detector.model_path = "weights/hockey_yolo.pt"
    config.detector.half_precision = True
    
    # Disable masks for speed
    config.mask.enabled = False
    config.tracker.use_masks = False
    config.tracker.use_reid = False  # Simpler matching
    
    config.visualization.show_masks = False
    
    pipeline = HockeyPipeline(config)
    results = pipeline.process_video(
        "hockey_game.mp4",
        "hockey_tracked_fast.mp4"
    )
    
    avg_fps = 1.0 / np.mean(pipeline.frame_times)
    print(f"Fast mode FPS: {avg_fps:.1f}")


def example_6_custom_classes():
    """
    Example 6: Custom Detection Classes
    """
    print("\n" + "="*60)
    print("Example 6: Custom Detection Classes")
    print("="*60)
    
    from hockey_mcbyte_tracker import (
        HockeyPipeline, PipelineConfig,
        HOCKEY_CLASS_COLORS
    )
    
    config = PipelineConfig()
    config.detector.model_path = "weights/custom_hockey_model.pt"
    
    # Custom class configuration
    config.detector.track_classes = [0, 1, 2]  # Your class IDs
    config.detector.class_names = {
        0: "Home Player",
        1: "Away Player",
        2: "Puck"
    }
    
    # Custom colors
    custom_colors = {
        0: (255, 0, 0),    # Blue for home
        1: (0, 0, 255),    # Red for away
        2: (0, 255, 255),  # Yellow for puck
    }
    
    # Apply to visualizer after pipeline creation
    pipeline = HockeyPipeline(config)
    pipeline.visualizer.class_colors = custom_colors
    
    results = pipeline.process_video("hockey_game.mp4", "custom_tracked.mp4")


def example_7_compare_with_without_occlusion():
    """
    Example 7: Compare tracking with and without occlusion handling
    """
    print("\n" + "="*60)
    print("Example 7: Occlusion Handling Comparison")
    print("="*60)
    
    from hockey_mcbyte_tracker import HockeyPipeline, PipelineConfig
    
    # Track with full occlusion handling
    config_full = PipelineConfig()
    config_full.detector.model_path = "weights/hockey_yolo.pt"
    config_full.mask.enabled = True
    config_full.tracker.use_masks = True
    config_full.tracker.use_appearance = True
    config_full.tracker.use_reid = True
    config_full.tracker.track_buffer = 90  # 3 seconds
    
    pipeline_full = HockeyPipeline(config_full)
    results_full = pipeline_full.process_video(
        "hockey_game.mp4",
        "tracked_full_occlusion.mp4"
    )
    
    # Track without occlusion handling (simple IoU only)
    config_simple = PipelineConfig()
    config_simple.detector.model_path = "weights/hockey_yolo.pt"
    config_simple.mask.enabled = False
    config_simple.tracker.use_masks = False
    config_simple.tracker.use_appearance = False
    config_simple.tracker.use_reid = False
    config_simple.tracker.track_buffer = 15  # Short buffer
    
    pipeline_simple = HockeyPipeline(config_simple)
    results_simple = pipeline_simple.process_video(
        "hockey_game.mp4",
        "tracked_simple.mp4"
    )
    
    # Compare results
    print(f"\nFull occlusion handling: {pipeline_full.tracker.tracker._count} unique IDs")
    print(f"Simple tracking: {pipeline_simple.tracker.tracker._count} unique IDs")
    print(f"Fewer IDs = better identity preservation")


def main():
    """Run all examples (with dummy video check)."""
    
    print("="*60)
    print("Hockey Tracker Usage Examples")
    print("="*60)
    print("\nThese examples demonstrate all features of the")
    print("occlusion-robust hockey tracking system.")
    print("\nTo run: Ensure you have:")
    print("  1. weights/hockey_yolo.pt (your YOLO model)")
    print("  2. weights/sam_vit_b_01ec64.pth (SAM weights)")
    print("  3. weights/cutie-base-mega.pth (CUTIE weights)")
    print("  4. hockey_game.mp4 (input video)")
    
    # Check if video exists before running
    if not Path("hockey_game.mp4").exists():
        print("\n[WARNING] hockey_game.mp4 not found.")
        print("These are reference examples - modify paths for your data.")
        return
    
    # Run examples
    # Uncomment the ones you want to run
    
    # example_1_quick_start()
    # example_2_full_configuration()
    # example_3_frame_by_frame()
    # example_4_extract_analytics()
    # example_5_without_masks()
    # example_6_custom_classes()
    # example_7_compare_with_without_occlusion()


if __name__ == "__main__":
    main()
