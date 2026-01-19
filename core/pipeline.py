"""
Hockey Tracking Pipeline
========================
Main pipeline combining detection, tracking, and mask propagation.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Generator, Any
from dataclasses import dataclass
import time


@dataclass
class FrameResult:
    """Result for a single frame."""
    frame_id: int
    tracks: List
    image: np.ndarray
    processing_time: float
    mask: Optional[np.ndarray] = None
    
    @property
    def num_tracks(self) -> int:
        return len(self.tracks)


class VideoReader:
    """Video/image sequence reader."""
    
    def __init__(self, source: str):
        """
        Initialize video reader.
        
        Args:
            source: Path to video file or image directory
        """
        self.source = Path(source)
        self.is_video = self.source.is_file()
        self.frame_id = 0
        
        if self.is_video:
            self.cap = cv2.VideoCapture(str(self.source))
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            self.images = []
            for ext in extensions:
                self.images.extend(sorted(self.source.glob(ext)))
            self.total_frames = len(self.images)
            self.fps = 30  # Default
            
            if self.total_frames > 0:
                img = cv2.imread(str(self.images[0]))
                self.height, self.width = img.shape[:2]
            else:
                self.width, self.height = 1920, 1080
    
    def __iter__(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Iterate over frames."""
        self.frame_id = 0
        
        if self.is_video:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.frame_id += 1
                yield self.frame_id, frame
        else:
            for img_path in self.images:
                self.frame_id += 1
                frame = cv2.imread(str(img_path))
                if frame is not None:
                    yield self.frame_id, frame
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read single frame."""
        if self.is_video:
            ret, frame = self.cap.read()
            if ret:
                self.frame_id += 1
            return ret, frame
        else:
            if self.frame_id < len(self.images):
                frame = cv2.imread(str(self.images[self.frame_id]))
                self.frame_id += 1
                return frame is not None, frame
            return False, None
    
    def release(self) -> None:
        """Release resources."""
        if self.is_video and hasattr(self, 'cap'):
            self.cap.release()
    
    def __len__(self) -> int:
        return self.total_frames


class VideoWriter:
    """Video output writer."""
    
    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float = 30.0,
        codec: str = "mp4v"
    ):
        """
        Initialize video writer.
        
        Args:
            output_path: Output video path
            width: Frame width
            height: Frame height
            fps: Frame rate
            codec: Video codec
        """
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        self.frame_count = 0
    
    def write(self, frame: np.ndarray) -> None:
        """Write frame to video."""
        self.writer.write(frame)
        self.frame_count += 1
    
    def release(self) -> None:
        """Release writer."""
        self.writer.release()


class HockeyTrackingPipeline:
    """
    Complete hockey tracking pipeline.
    
    Integrates:
    - Custom YOLO detection
    - McByte tracking with mask propagation
    - Visualization
    """
    
    def __init__(self, config=None):
        """
        Initialize pipeline.
        
        Args:
            config: PipelineConfig object (uses defaults if None)
        """
        from configs.config import PipelineConfig
        self.config = config or PipelineConfig()
        
        # Components (lazy initialization)
        self._detector = None
        self._tracker = None
        self._mask_propagator = None
        self._visualizer = None
        
        # State
        self.frame_id = 0
        self.initialized = False
    
    @property
    def detector(self):
        """Lazy load detector."""
        if self._detector is None:
            from detectors.yolo_detector import create_detector
            self._detector = create_detector(self.config.detector)
            self._detector.warmup()
        return self._detector
    
    @property
    def tracker(self):
        """Lazy load tracker."""
        if self._tracker is None:
            from tracking.mcbyte_tracker import McByteTracker
            self._tracker = McByteTracker(
                track_thresh=self.config.tracker.track_thresh,
                track_buffer=self.config.tracker.track_buffer,
                match_thresh=self.config.tracker.match_thresh,
                low_thresh=self.config.tracker.low_thresh,
                new_track_thresh=self.config.tracker.new_track_thresh,
                frame_rate=self.config.tracker.frame_rate,
                use_gmc=self.config.tracker.use_gmc,
                gmc_method=self.config.tracker.gmc_method,
            )
        return self._tracker
    
    @property
    def mask_propagator(self):
        """Lazy load mask propagator."""
        if self._mask_propagator is None:
            from mask_propagation.mask_propagator import create_mask_propagator
            self._mask_propagator = create_mask_propagator(self.config.mask)
        return self._mask_propagator
    
    @property
    def visualizer(self):
        """Lazy load visualizer."""
        if self._visualizer is None:
            from utils.visualization import Visualizer
            self._visualizer = Visualizer(
                show_boxes=self.config.visualization.show_boxes,
                show_ids=self.config.visualization.show_ids,
                show_masks=self.config.visualization.show_masks,
                show_trajectories=self.config.visualization.show_trajectories,
                trajectory_length=self.config.visualization.trajectory_length,
                class_colors=self.config.visualization.class_colors,
                class_names=self.config.detector.class_names,
            )
        return self._visualizer
    
    def process_frame(
        self,
        frame: np.ndarray,
        use_masks: bool = True
    ) -> FrameResult:
        """
        Process single frame through pipeline.
        
        Args:
            frame: BGR image
            use_masks: Enable mask propagation
            
        Returns:
            FrameResult with tracks and timing
        """
        start_time = time.time()
        self.frame_id += 1
        
        # Detection
        detections = self.detector.detect(frame)
        
        # Get mask info for association
        mask_info = None
        if use_masks and self.frame_id > 1:
            mask_state = self.mask_propagator.propagate(frame)
            if mask_state.prediction is not None:
                mask_info = {
                    'mask': mask_state.prediction,
                    'track_to_mask': mask_state.track_to_mask,
                }
        
        # Tracking
        tracks = self.tracker.update(detections, frame, mask_info)
        
        # Initialize/update masks for new tracks
        if use_masks:
            if self.frame_id == 1:
                # Initialize masks
                bboxes = [t.tlwh for t in tracks]
                track_ids = [t.track_id for t in tracks]
                if len(bboxes) > 0:
                    self.mask_propagator.initialize_masks(frame, bboxes, track_ids)
            else:
                # Add masks for new tracks
                new_tracks = [t for t in tracks if t.tracklet_len == 0]
                if len(new_tracks) > 0:
                    new_bboxes = [t.tlwh for t in new_tracks]
                    new_ids = [t.track_id for t in new_tracks]
                    all_bboxes = [t.tlwh for t in tracks]
                    self.mask_propagator.add_masks(
                        frame, new_bboxes, new_ids, all_bboxes
                    )
        
        # Get final mask for visualization
        mask = None
        if use_masks and hasattr(self.mask_propagator, 'state'):
            mask = self.mask_propagator.state.prediction
        
        processing_time = time.time() - start_time
        
        return FrameResult(
            frame_id=self.frame_id,
            tracks=tracks,
            image=frame,
            processing_time=processing_time,
            mask=mask,
        )
    
    def visualize(self, result: FrameResult) -> np.ndarray:
        """
        Visualize tracking result.
        
        Args:
            result: FrameResult from process_frame
            
        Returns:
            Annotated frame
        """
        track_to_mask = None
        if hasattr(self.mask_propagator, 'state'):
            track_to_mask = self.mask_propagator.state.track_to_mask
        
        return self.visualizer.draw_tracks(
            result.image,
            result.tracks,
            result.mask,
            track_to_mask,
        )
    
    def run(
        self,
        input_source: str,
        output_path: Optional[str] = None,
        show_preview: bool = False,
        use_masks: bool = True,
        progress_callback=None,
    ) -> List[FrameResult]:
        """
        Run tracking on video/images.
        
        Args:
            input_source: Path to video or image directory
            output_path: Output video path (None to skip saving)
            show_preview: Display frames during processing
            use_masks: Enable mask propagation
            progress_callback: Optional callback(frame_id, total, result)
            
        Returns:
            List of FrameResults
        """
        reader = VideoReader(input_source)
        results = []
        
        # Setup output
        writer = None
        if output_path:
            writer = VideoWriter(
                output_path,
                reader.width, reader.height,
                reader.fps
            )
        
        print(f"[Pipeline] Processing {len(reader)} frames...")
        print(f"[Pipeline] Resolution: {reader.width}x{reader.height}")
        print(f"[Pipeline] FPS: {reader.fps}")
        
        try:
            for frame_id, frame in reader:
                # Process
                result = self.process_frame(frame, use_masks)
                results.append(result)
                
                # Visualize
                vis_frame = self.visualize(result)
                
                # Write output
                if writer:
                    writer.write(vis_frame)
                
                # Preview
                if show_preview:
                    cv2.imshow('Tracking', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Progress
                if progress_callback:
                    progress_callback(frame_id, len(reader), result)
                elif frame_id % 10 == 0:
                    avg_time = np.mean([r.processing_time for r in results[-10:]])
                    print(f"  Frame {frame_id}/{len(reader)} | "
                          f"Tracks: {result.num_tracks} | "
                          f"Time: {avg_time*1000:.1f}ms")
        
        finally:
            reader.release()
            if writer:
                writer.release()
            if show_preview:
                cv2.destroyAllWindows()
        
        # Summary
        total_time = sum(r.processing_time for r in results)
        avg_fps = len(results) / total_time if total_time > 0 else 0
        
        print(f"\n[Pipeline] Complete!")
        print(f"[Pipeline] Processed {len(results)} frames in {total_time:.2f}s")
        print(f"[Pipeline] Average: {avg_fps:.1f} FPS")
        
        if output_path:
            print(f"[Pipeline] Output saved to: {output_path}")
        
        return results
    
    def reset(self) -> None:
        """Reset pipeline state."""
        self.frame_id = 0
        if self._tracker:
            from tracking.track import Track
            Track.reset_id()
            self._tracker = None
        if self._mask_propagator:
            self._mask_propagator.reset()
        if self._visualizer:
            self._visualizer.reset_histories()
    
    def get_mot_results(self, results: List[FrameResult]) -> str:
        """
        Convert results to MOT format string.
        
        Format: frame, id, x, y, w, h, conf, class, -1, -1
        
        Args:
            results: List of FrameResults
            
        Returns:
            MOT format string
        """
        lines = []
        for result in results:
            for track in result.tracks:
                x, y, w, h = track.tlwh
                line = (f"{result.frame_id},{track.track_id},"
                       f"{x:.2f},{y:.2f},{w:.2f},{h:.2f},"
                       f"{track.score:.2f},{track.class_id},-1,-1")
                lines.append(line)
        
        return '\n'.join(lines)
    
    def save_mot_results(
        self,
        results: List[FrameResult],
        output_path: str
    ) -> None:
        """Save results in MOT format."""
        mot_str = self.get_mot_results(results)
        with open(output_path, 'w') as f:
            f.write(mot_str)
        print(f"[Pipeline] MOT results saved to: {output_path}")


def run_tracking(
    input_path: str,
    model_path: str,
    output_path: Optional[str] = None,
    track_classes: Optional[List[int]] = None,
    show_preview: bool = False,
    use_masks: bool = True,
) -> List[FrameResult]:
    """
    Convenience function to run tracking with minimal setup.
    
    Args:
        input_path: Video or image directory path
        model_path: YOLO model weights path
        output_path: Output video path (None to skip)
        track_classes: Class IDs to track (None for defaults)
        show_preview: Show preview window
        use_masks: Enable mask propagation
        
    Returns:
        List of FrameResults
    """
    from configs.config import PipelineConfig
    
    # Setup config
    config = PipelineConfig()
    config.detector.model_path = model_path
    
    if track_classes:
        config.detector.track_classes = track_classes
    
    # Create and run pipeline
    pipeline = HockeyTrackingPipeline(config)
    results = pipeline.run(
        input_path,
        output_path,
        show_preview=show_preview,
        use_masks=use_masks,
    )
    
    return results
