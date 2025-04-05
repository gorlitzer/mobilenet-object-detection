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

    def process_video_stream(self, source: int = 0) -> None:
        """
        Process video stream from camera or video file.
        
        Args:
            source: Camera index (int) or video file path (str)
        """
        # Check if we're on a Raspberry Pi
        is_raspberry_pi = self._is_raspberry_pi()
        
        # Determine if we should use picamera2
        use_picamera = USE_PICAMERA and is_raspberry_pi
        
        if use_picamera:
            logger.info("Using picamera2 for Raspberry Pi camera")
            self._process_picamera_stream()
        else:
            logger.info("Using OpenCV for video capture")
            self._process_opencv_stream(source)
    
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
            
    def _process_picamera_stream(self) -> None:
        """Process video stream from Raspberry Pi camera using picamera2."""
        try:
            # Try to import picamera2
            try:
                from picamera2 import Picamera2
                import time
                logger.info("Successfully imported picamera2 module")
            except ImportError:
                logger.error("Picamera2 module not found. Please install it on your Raspberry Pi.")
                logger.error("Run the install_picamera2.sh script in the project directory:")
                logger.error("chmod +x install_picamera2.sh")
                logger.error("./install_picamera2.sh")
                logger.error("Or manually install with: sudo apt install -y python3-picamera2")
                logger.error("After installation, you may need to reboot your Raspberry Pi.")
                raise
            
            # Initialize picamera2
            try:
                picam2 = Picamera2()
                logger.info("Picamera2 initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize picamera2: {e}")
                logger.error("Make sure your camera is properly connected and enabled.")
                logger.error("Run: sudo raspi-config and enable the camera in Interface Options.")
                raise
            
            # Configure camera
            try:
                config = picam2.create_preview_configuration(main=CAMERA_CONFIG['main'])
                picam2.configure(config)
                logger.info("Picamera2 configured successfully")
            except Exception as e:
                logger.error(f"Failed to configure picamera2: {e}")
                picam2.close()
                raise
            
            # Start camera
            try:
                picam2.start()
                logger.info("Picamera2 started successfully")
            except Exception as e:
                logger.error(f"Failed to start picamera2: {e}")
                picam2.close()
                raise
            
            try:
                while True:
                    # Capture frame
                    try:
                        frame = picam2.capture_array()
                    except Exception as e:
                        logger.error(f"Failed to capture frame: {e}")
                        continue
                    
                    # Convert from RGB to BGR for OpenCV
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Process frame
                    processed_frame, detections = self.detect(frame)
                    
                    # Display frame
                    cv2.imshow('Road Sign Detection', processed_frame)
                    
                    # Break loop on 'q' press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
            finally:
                picam2.stop()
                cv2.destroyAllWindows()
                
        except Exception as e:
            logger.error(f"Error with picamera2: {e}")
            logger.error("Falling back to OpenCV capture")
            self._process_opencv_stream(0)
            
    def _process_opencv_stream(self, source: int = 0) -> None:
        """Process video stream using OpenCV."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {source}")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break

                # Process frame
                processed_frame, detections = self.detect(frame)
                
                # Display frame
                cv2.imshow('Road Sign Detection', processed_frame)
                
                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows() 