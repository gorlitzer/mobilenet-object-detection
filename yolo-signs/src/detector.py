import cv2
import numpy as np
from ultralytics import YOLO
import time
from typing import Tuple, List, Dict, Optional
import logging
import os
import platform

from config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    ROAD_SIGN_CLASSES,
    DISPLAY_CONFIDENCE,
    DISPLAY_BOUNDING_BOX,
    BOUNDING_BOX_COLOR,
    BOUNDING_BOX_THICKNESS,
    TEXT_COLOR,
    TEXT_THICKNESS,
    TEXT_SCALE,
    USE_PICAMERA,
    CAMERA_CONFIG,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RoadSignDetector:
    def __init__(self):
        """Initialize the YOLO model for road sign detection."""
        try:
            self.model = YOLO(MODEL_PATH)
            logger.info(f"Loaded YOLO model from {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect road signs in the given frame.
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            Tuple containing:
            - Processed frame with detections drawn
            - List of detection dictionaries with class, confidence, and bounding box
        """
        # Run YOLO detection
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD)[0]
        
        # Process detections
        detections = []
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            class_name = results.names[int(class_id)]
            
            # Only process road sign related classes
            if class_name in ROAD_SIGN_CLASSES:
                detection = {
                    'class': class_name,
                    'confidence': score,
                    'bbox': (int(x1), int(y1), int(x2), int(y2))
                }
                detections.append(detection)
                
                # Draw bounding box and label
                if DISPLAY_BOUNDING_BOX:
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        BOUNDING_BOX_COLOR,
                        BOUNDING_BOX_THICKNESS
                    )
                    
                    # Prepare label text
                    label = f"{class_name}"
                    if DISPLAY_CONFIDENCE:
                        label += f" {score:.2f}"
                        
                    # Draw label background
                    (label_width, label_height), _ = cv2.getTextSize(
                        label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        TEXT_SCALE,
                        TEXT_THICKNESS
                    )
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1) - label_height - 10),
                        (int(x1) + label_width, int(y1)),
                        BOUNDING_BOX_COLOR,
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        frame,
                        label,
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        TEXT_SCALE,
                        TEXT_COLOR,
                        TEXT_THICKNESS
                    )
        
        return frame, detections

    def process_video_stream(self, source: int = 0, headless: bool = False) -> None:
        """
        Process video stream from the given source.
        
        Args:
            source: Camera index or video file path
            headless: If True, run without displaying frames (useful for servers without display)
        """
        if self._is_raspberry_pi() and USE_PICAMERA:
            self._process_picamera_stream(headless=headless)
        else:
            self._process_opencv_stream(source, headless=headless)
    
    def _is_raspberry_pi(self) -> bool:
        """Check if the current system is a Raspberry Pi."""
        # Check for Raspberry Pi hardware
        if os.path.exists('/proc/device-tree/model'):
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip('\0')
                if 'Raspberry Pi' in model:
                    return True
        
        # Check platform
        if platform.system() == 'Linux':
            # Additional check for Raspberry Pi OS
            if os.path.exists('/etc/rpi-issue'):
                return True
        
        return False
            
    def _process_picamera_stream(self, headless: bool = False) -> None:
        """
        Process video stream from picamera2.
        
        Args:
            headless: If True, run without displaying frames (useful for servers without display)
        """
        try:
            from picamera2 import Picamera2
            import time
            
            # Initialize picamera2
            picam2 = Picamera2()
            
            # Configure camera
            config = picam2.create_preview_configuration(main=CAMERA_CONFIG['main'])
            picam2.configure(config)
            
            # Start camera
            picam2.start()
            logger.info("Picamera2 started successfully")
            
            # Initialize FPS counter
            fps_counter = FPSCounter()
            
            try:
                while True:
                    # Capture frame
                    frame = picam2.capture_array()
                    
                    # Convert from RGB to BGR (OpenCV format)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Process frame
                    processed_frame, detections = self.detect(frame)
                    
                    # Update FPS counter
                    fps = fps_counter.update()
                    processed_frame = draw_fps(processed_frame, fps)
                    
                    # Save detections
                    if detections:
                        save_frame(processed_frame, detections[0])
                    
                    # Display frame if not in headless mode
                    if not headless:
                        cv2.imshow('Road Sign Detection', processed_frame)
                        
                        # Break loop on 'q' press
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        # In headless mode, just print FPS periodically
                        if fps_counter.frame_count % 30 == 0:  # Print every 30 frames
                            logger.info(f"FPS: {fps:.2f}")
                    
            finally:
                picam2.stop()
                if not headless:
                    cv2.destroyAllWindows()
                logger.info("Picamera2 stopped")
                
        except ImportError:
            logger.error("picamera2 not available. Please install it for Raspberry Pi support.")
            raise
        except Exception as e:
            logger.error(f"Error processing picamera2 stream: {e}")
            raise
            
    def _process_opencv_stream(self, source: int = 0, headless: bool = False) -> None:
        """
        Process video stream using OpenCV.
        
        Args:
            source: Camera index or video file path
            headless: If True, run without displaying frames (useful for servers without display)
        """
        # Process video stream
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {source}")
            return
            
        # Initialize FPS counter
        fps_counter = FPSCounter()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break
                    
                # Process frame
                processed_frame, detections = self.detect(frame)
                
                # Update FPS counter
                fps = fps_counter.update()
                processed_frame = draw_fps(processed_frame, fps)
                
                # Save detections
                if detections:
                    save_frame(processed_frame, detections[0])
                
                # Display frame if not in headless mode
                if not headless:
                    cv2.imshow('Road Sign Detection', processed_frame)
                    
                    # Break loop on 'q' press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    # In headless mode, just print FPS periodically
                    if fps_counter.frame_count % 30 == 0:  # Print every 30 frames
                        logger.info(f"FPS: {fps:.2f}")
                    
        finally:
            cap.release()
            if not headless:
                cv2.destroyAllWindows()
            logger.info("Road sign detection stopped") 