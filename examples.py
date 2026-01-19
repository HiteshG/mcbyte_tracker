#!/usr/bin/env python
"""
Hockey McByte Tracker - Usage Examples
======================================
Demonstrates various ways to use the tracking system.
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def example_basic_usage():
    """
    Example 1: Basic tracking with minimal code.
    """
    from core.pipeline import run_tracking
    
    # Track players, puck, referee, goaltender
    results = run_tracking(
        input_path="path/to/hockey_game.mp4",
        model_path="weights/hockey_yolo.pt",
        output_path="output/tracked_game.mp4",
        track_classes=[3, 4, 5, 6],  # Goaltender, Player, Puck, Referee
        use_masks=True,
    )
    
    # Print summary
    for result in results[:5]:
        print(f"Frame {result.frame_id}: {result.num_tracks} tracks")


def example_custom_config():
    """
    Example 2: Custom configuration for different scenarios.
    """
    from configs.config import PipelineConfig
    from core.pipeline import HockeyTrackingPipeline
    
    # Create custom configuration
    config = PipelineConfig()
    
    # Detection settings
    config.detector.model_path = "weights/hockey_yolo.pt"
    config.detector.confidence_threshold = 0.3  # Lower for more detections
    config.detector.nms_threshold = 0.5
    config.detector.track_classes = [3, 4, 5, 6]  # Track these classes only
    config.detector.half_precision = True  # Use FP16 for speed
    
    # Tracker settings  
    config.tracker.track_thresh = 0.5  # High-confidence threshold
    config.tracker.track_buffer = 45  # Keep lost tracks longer
    config.tracker.match_thresh = 0.85
    config.tracker.use_gmc = True  # Camera motion compensation
    
    # Visualization settings
    config.visualization.show_boxes = True
    config.visualization.show_ids = True
    config.visualization.show_trajectories = True
    config.visualization.trajectory_length = 60  # 2 seconds at 30fps
    
    # Create and run pipeline
    pipeline = HockeyTrackingPipeline(config)
    results = pipeline.run(
        input_source="path/to/video.mp4",
        output_path="output/tracked.mp4",
        use_masks=True,
    )


def example_frame_by_frame():
    """
    Example 3: Frame-by-frame processing for custom integration.
    """
    import cv2
    from configs.config import PipelineConfig
    from core.pipeline import HockeyTrackingPipeline
    
    # Setup
    config = PipelineConfig()
    config.detector.model_path = "weights/hockey_yolo.pt"
    config.detector.track_classes = [3, 4, 5, 6]
    
    pipeline = HockeyTrackingPipeline(config)
    
    # Open video
    cap = cv2.VideoCapture("path/to/video.mp4")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        result = pipeline.process_frame(frame, use_masks=False)
        
        # Get track information
        for track in result.tracks:
            track_id = track.track_id
            bbox = track.tlwh  # [x, y, width, height]
            class_id = track.class_id
            confidence = track.score
            
            print(f"Track {track_id}: class={class_id}, conf={confidence:.2f}")
        
        # Custom visualization
        vis_frame = pipeline.visualize(result)
        cv2.imshow('Tracking', vis_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def example_tracking_only():
    """
    Example 4: Using tracker with external detections.
    """
    from tracking.mcbyte_tracker import McByteTracker
    import numpy as np
    
    # Initialize tracker
    tracker = McByteTracker(
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        frame_rate=30,
    )
    
    # Simulate detections (normally from your detector)
    detections = [
        # (tlwh, score, class_id)
        ([100, 200, 50, 80], 0.95, 4),  # Player
        ([300, 250, 45, 75], 0.88, 4),  # Player
        ([500, 180, 20, 15], 0.72, 5),  # Puck
    ]
    
    # Update tracker
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)  # Dummy frame
    tracks = tracker.update(detections, frame)
    
    for track in tracks:
        print(f"Track ID: {track.track_id}")
        print(f"  Position: {track.tlwh}")
        print(f"  Score: {track.score:.2f}")
        print(f"  Class: {track.class_id}")


def example_export_mot():
    """
    Example 5: Export tracking results in MOT format.
    """
    from configs.config import PipelineConfig
    from core.pipeline import HockeyTrackingPipeline
    
    config = PipelineConfig()
    config.detector.model_path = "weights/hockey_yolo.pt"
    config.detector.track_classes = [3, 4, 5, 6]
    
    pipeline = HockeyTrackingPipeline(config)
    results = pipeline.run(
        input_source="path/to/video.mp4",
        output_path="output/tracked.mp4",
    )
    
    # Save in MOT format
    pipeline.save_mot_results(results, "output/tracking_results.txt")
    
    # Or get as string
    mot_string = pipeline.get_mot_results(results)
    print(mot_string[:500])  # Print first 500 chars


def example_performance_optimization():
    """
    Example 6: Optimized settings for real-time processing.
    """
    from configs.config import PipelineConfig
    from core.pipeline import HockeyTrackingPipeline
    
    config = PipelineConfig()
    
    # Use smaller model for speed (if available)
    config.detector.model_path = "weights/hockey_yolo.pt"
    config.detector.half_precision = True  # FP16 inference
    config.detector.confidence_threshold = 0.35  # Slightly higher to reduce detections
    
    # Disable masks for maximum speed
    # Masks add accuracy but reduce FPS significantly
    
    # Reduce track buffer
    config.tracker.track_buffer = 15
    
    # Create pipeline
    pipeline = HockeyTrackingPipeline(config)
    
    # Run without masks for speed
    results = pipeline.run(
        input_source="path/to/video.mp4",
        output_path="output/fast_tracked.mp4",
        use_masks=False,  # Disable for ~3x speed
    )


def example_hockey_analysis():
    """
    Example 7: Extract hockey-specific analytics.
    """
    from configs.config import PipelineConfig
    from core.pipeline import HockeyTrackingPipeline
    import numpy as np
    
    config = PipelineConfig()
    config.detector.model_path = "weights/hockey_yolo.pt"
    config.detector.track_classes = [3, 4, 5, 6]
    
    # Define class mapping
    CLASS_NAMES = {
        3: "Goaltender",
        4: "Player", 
        5: "Puck",
        6: "Referee"
    }
    
    pipeline = HockeyTrackingPipeline(config)
    results = pipeline.run(
        input_source="path/to/video.mp4",
        use_masks=False,
    )
    
    # Analyze results
    track_data = {}  # track_id -> list of positions
    
    for result in results:
        for track in result.tracks:
            if track.track_id not in track_data:
                track_data[track.track_id] = {
                    'class_id': track.class_id,
                    'class_name': CLASS_NAMES.get(track.class_id, "Unknown"),
                    'positions': [],
                }
            x, y, w, h = track.tlwh
            center = (x + w/2, y + h/2)
            track_data[track.track_id]['positions'].append(center)
    
    # Calculate statistics
    print("\n=== Hockey Tracking Analysis ===\n")
    
    puck_tracks = [t for t in track_data.values() if t['class_id'] == 5]
    player_tracks = [t for t in track_data.values() if t['class_id'] == 4]
    
    print(f"Total player tracks: {len(player_tracks)}")
    print(f"Total puck tracks: {len(puck_tracks)}")
    
    # Calculate player distances
    for track_id, data in track_data.items():
        if data['class_id'] == 4:  # Player
            positions = np.array(data['positions'])
            if len(positions) > 1:
                diffs = np.diff(positions, axis=0)
                distances = np.sqrt(np.sum(diffs**2, axis=1))
                total_distance = np.sum(distances)
                print(f"Track {track_id} ({data['class_name']}): {total_distance:.1f} pixels traveled")


if __name__ == "__main__":
    print("Hockey McByte Tracker - Examples")
    print("================================")
    print()
    print("Available examples:")
    print("  1. example_basic_usage() - Simple one-liner tracking")
    print("  2. example_custom_config() - Custom configuration")
    print("  3. example_frame_by_frame() - Manual frame processing")
    print("  4. example_tracking_only() - Use tracker with external detections")
    print("  5. example_export_mot() - Export in MOT format")
    print("  6. example_performance_optimization() - Real-time settings")
    print("  7. example_hockey_analysis() - Extract analytics")
    print()
    print("Run any example function to see it in action.")
    print("Note: Adjust paths to your actual files before running.")
