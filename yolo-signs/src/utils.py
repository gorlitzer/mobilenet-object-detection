import cv2
import time
from typing import Optional
import logging
import os
from datetime import datetime
from config import DISPLAY_FPS

logger = logging.getLogger(__name__)

class FPSCounter:
    """Simple FPS counter for video processing."""
    
    def __init__(self, avg_frames: int = 30):
        self.avg_frames = avg_frames
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
    def update(self) -> float:
        """Update FPS calculation."""
        self.frame_count += 1
        
        if self.frame_count >= self.avg_frames:
            end_time = time.time()
            self.fps = self.frame_count / (end_time - self.start_time)
            self.frame_count = 0
            self.start_time = time.time()
            
        return self.fps

def draw_fps(frame: cv2.Mat, fps: float) -> cv2.Mat:
    """
    Draw FPS counter on frame.
    
    Args:
        frame: Input frame
        fps: Current FPS value
        
    Returns:
        Frame with FPS counter drawn
    """
    if DISPLAY_FPS:
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
    return frame

def save_frame(frame: cv2.Mat, detection: dict) -> Optional[str]:
    """
    Save frame with detection to file.
    
    Args:
        frame: Input frame
        detection: Detection information
        
    Returns:
        Path to saved file if successful, None otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs('data/detections', exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"data/detections/{detection['class']}_{timestamp}.jpg"
        
        # Save frame
        cv2.imwrite(filename, frame)
        logger.info(f"Saved detection frame to {filename}")
        
        return filename
        
    except Exception as e:
        logger.error(f"Failed to save detection frame: {e}")
        return None 