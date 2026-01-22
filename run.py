#!/usr/bin/env python3
"""
Hockey McByte Tracker - Command Line Interface
==============================================
Run tracking from command line.

Usage:
    python run.py input.mp4 -o output.mp4 -m weights/hockey_yolo.pt
    python run.py input.mp4 --fast  # Fast mode without masks
    python run.py input.mp4 --accurate  # Full accuracy mode
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Hockey McByte Tracker v2.0 - Occlusion-Robust Tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py game.mp4 -o tracked.mp4 -m weights/hockey_yolo.pt
  python run.py game.mp4 --fast --preview
  python run.py game.mp4 --accurate -m weights/hockey_yolo.pt
        """
    )
    
    parser.add_argument("input", help="Input video path")
    parser.add_argument("-o", "--output", help="Output video path (default: input_tracked.mp4)")
    parser.add_argument("-m", "--model", default="weights/hockey_yolo.pt",
                        help="YOLO model path (default: weights/hockey_yolo.pt)")
    
    # Mode presets
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--fast", action="store_true",
                           help="Fast mode: no masks, no ReID (~30 FPS)")
    mode_group.add_argument("--balanced", action="store_true", default=True,
                           help="Balanced mode: masks + histogram ReID (~15 FPS)")
    mode_group.add_argument("--accurate", action="store_true",
                           help="Accurate mode: full features (~5 FPS)")
    
    # Detection options
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Detection confidence threshold (default: 0.3)")
    parser.add_argument("--classes", type=int, nargs="+", default=[3, 4, 5, 6],
                        help="Class IDs to track (default: 3 4 5 6)")
    
    # Tracker options
    parser.add_argument("--buffer", type=int, default=60,
                        help="Track buffer in frames (default: 60)")
    parser.add_argument("--no-masks", action="store_true",
                        help="Disable mask propagation")
    parser.add_argument("--no-reid", action="store_true",
                        help="Disable re-identification")
    
    # Output options
    parser.add_argument("--preview", action="store_true",
                        help="Show live preview during processing")
    parser.add_argument("--mot", action="store_true",
                        help="Save results in MOT format")
    parser.add_argument("--device", default="cuda",
                        help="Device: cuda or cpu (default: cuda)")
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Set output path
    if args.output is None:
        input_stem = Path(args.input).stem
        args.output = f"{input_stem}_tracked.mp4"
    
    # Import here to avoid slow startup for --help
    from hockey_mcbyte_tracker import HockeyPipeline, PipelineConfig
    
    # Create config based on mode
    config = PipelineConfig()
    
    if args.fast:
        print("[Mode] Fast - Optimized for speed")
        config.mask.enabled = False
        config.tracker.use_masks = False
        config.tracker.use_reid = False
        config.tracker.track_buffer = 15
        config.tracker.motion_weight = 0.7
        config.tracker.appearance_weight = 0.3
        config.tracker.mask_weight = 0.0
    elif args.accurate:
        print("[Mode] Accurate - Full feature set")
        config.mask.enabled = True
        config.tracker.use_masks = True
        config.tracker.use_reid = True
        config.tracker.track_buffer = 90
        config.tracker.motion_weight = 0.3
        config.tracker.appearance_weight = 0.35
        config.tracker.mask_weight = 0.35
    else:
        print("[Mode] Balanced - Default configuration")
    
    # Apply command line overrides
    config.detector.model_path = args.model
    config.detector.confidence_threshold = args.conf
    config.detector.track_classes = args.classes
    config.detector.device = args.device
    config.tracker.track_buffer = args.buffer
    
    if args.no_masks:
        config.mask.enabled = False
        config.tracker.use_masks = False
    
    if args.no_reid:
        config.tracker.use_reid = False
    
    config.save_mot = args.mot
    
    # Run pipeline
    print(f"\n[Input]  {args.input}")
    print(f"[Output] {args.output}")
    print(f"[Model]  {args.model}")
    print(f"[Classes] {args.classes}")
    print()
    
    try:
        pipeline = HockeyPipeline(config)
        results = pipeline.process_video(
            args.input,
            args.output,
            show_progress=True,
            preview=args.preview
        )
        
        if args.mot:
            mot_path = Path(args.output).stem + "_mot.txt"
            pipeline.save_mot_results(mot_path)
        
        print(f"\n[Complete] Processed {len(results)} frames")
        print(f"[Output] Saved to {args.output}")
        
    except KeyboardInterrupt:
        print("\n[Interrupted] Processing stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
