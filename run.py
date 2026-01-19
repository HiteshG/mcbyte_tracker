#!/usr/bin/env python
"""
Hockey McByte Tracker - Quick Run Script
=========================================
Simple script to run tracking on a video or image sequence.

Usage:
    python run.py --input video.mp4 --model weights/hockey_yolo.pt --output output.mp4
    python run.py --input frames/ --model weights/hockey_yolo.pt --output output.mp4 --no-masks
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="Hockey McByte Tracker - Multi-Object Tracking for Hockey",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Track with masks
    python run.py --input game.mp4 --model weights/hockey_yolo.pt --output tracked.mp4
    
    # Track without masks (faster)
    python run.py --input game.mp4 --model weights/hockey_yolo.pt --output tracked.mp4 --no-masks
    
    # Track specific classes only
    python run.py --input game.mp4 --model weights/hockey_yolo.pt --output tracked.mp4 --classes 3 4 5 6
    
    # Show preview during processing
    python run.py --input game.mp4 --model weights/hockey_yolo.pt --preview
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input video or image directory"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to YOLO model weights (.pt file)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output video path (default: input_tracked.mp4)"
    )
    parser.add_argument(
        "--classes", "-c",
        nargs="+",
        type=int,
        default=[3, 4, 5, 6],
        help="Class IDs to track (default: 3 4 5 6 = Goaltender, Player, Puck, Referee)"
    )
    parser.add_argument(
        "--no-masks",
        action="store_true",
        help="Disable mask propagation (faster but less accurate)"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show preview window during processing"
    )
    parser.add_argument(
        "--save-mot",
        default=None,
        help="Save MOT format results to this path"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--track-thresh",
        type=float,
        default=0.5,
        help="Tracking confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--track-buffer",
        type=int,
        default=30,
        help="Frames to keep lost tracks (default: 30)"
    )
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input not found: {args.input}")
        sys.exit(1)
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)
    
    # Set default output
    if args.output is None:
        if input_path.is_file():
            args.output = str(input_path.with_stem(input_path.stem + "_tracked"))
        else:
            args.output = str(input_path / "output_tracked.mp4")
    
    # Import and configure
    from configs.config import PipelineConfig
    from core.pipeline import HockeyTrackingPipeline
    
    print("=" * 60)
    print("Hockey McByte Tracker")
    print("=" * 60)
    print(f"Input:   {args.input}")
    print(f"Model:   {args.model}")
    print(f"Output:  {args.output}")
    print(f"Classes: {args.classes}")
    print(f"Masks:   {'Enabled' if not args.no_masks else 'Disabled'}")
    print("=" * 60)
    
    # Create configuration
    config = PipelineConfig()
    config.detector.model_path = str(model_path)
    config.detector.track_classes = args.classes
    config.detector.confidence_threshold = args.confidence
    config.tracker.track_thresh = args.track_thresh
    config.tracker.track_buffer = args.track_buffer
    
    # Hockey class names
    config.detector.class_names = {
        0: "Center Ice",
        1: "Faceoff",
        2: "Goalpost",
        3: "Goaltender",
        4: "Player",
        5: "Puck",
        6: "Referee"
    }
    
    # Create and run pipeline
    pipeline = HockeyTrackingPipeline(config)
    
    results = pipeline.run(
        input_source=str(input_path),
        output_path=args.output,
        show_preview=args.preview,
        use_masks=not args.no_masks,
    )
    
    # Save MOT results if requested
    if args.save_mot:
        pipeline.save_mot_results(results, args.save_mot)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
