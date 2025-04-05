import argparse
import logging
import cv2
import os
from detector import RoadSignDetector
from utils import FPSCounter, TelegramNotifier, draw_fps, save_frame
from config import DEFAULT_CAMERA_ID, USE_PICAMERA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Road Sign Detection using YOLO')
    parser.add_argument(
        '--source',
        type=str,
        default=str(DEFAULT_CAMERA_ID),
        help='Camera index (int) or video file path (str). Ignored if USE_PICAMERA is True.'
    )
    parser.add_argument(
        '--use-picamera',
        action='store_true',
        help='Force using picamera2 (Raspberry Pi camera)'
    )
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    args = parse_args()
    
    # Initialize components
    detector = RoadSignDetector()
    fps_counter = FPSCounter()
    telegram_notifier = TelegramNotifier()
    
    # Determine if we should use picamera2
    use_picamera = args.use_picamera or USE_PICAMERA
    
    if use_picamera:
        logger.info("Using picamera2 for Raspberry Pi camera")
        detector.process_video_stream()
    else:
        # Convert source to int if it's a camera index
        try:
            source = int(args.source)
        except ValueError:
            source = args.source
            
        # Check if source exists
        if isinstance(source, str) and not os.path.exists(source):
            logger.error(f"Source file does not exist: {source}")
            return
            
        logger.info(f"Starting road sign detection from source: {source}")
        
        # Process video stream
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
                processed_frame, detections = detector.detect(frame)
                
                # Update FPS counter
                fps = fps_counter.update()
                processed_frame = draw_fps(processed_frame, fps)
                
                # Handle detections
                for detection in detections:
                    # Save frame with detection
                    save_frame(processed_frame, detection)
                    
                    # Send Telegram notification
                    telegram_notifier.send_notification(detection)
                
                # Display frame
                cv2.imshow('Road Sign Detection', processed_frame)
                
                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Road sign detection stopped")

if __name__ == '__main__':
    main() 