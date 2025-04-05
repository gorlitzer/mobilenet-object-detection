import argparse
import logging
import cv2
import os
from detector import RoadSignDetector
from utils import FPSCounter, draw_fps, save_frame
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
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run in headless mode without displaying frames (useful for servers without display)'
    )
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    args = parse_args()
    
    # Initialize components
    detector = RoadSignDetector()
    fps_counter = FPSCounter()
    
    # Determine if we should use picamera2
    use_picamera = args.use_picamera or USE_PICAMERA
    
    if use_picamera:
        logger.info("Using picamera2 for Raspberry Pi camera")
        detector.process_video_stream(headless=args.headless)
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
                
                # Save detections
                if detections:
                    save_frame(processed_frame, detections[0])
                
                # Display frame if not in headless mode
                if not args.headless:
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
            if not args.headless:
                cv2.destroyAllWindows()
            logger.info("Road sign detection stopped")

if __name__ == '__main__':
    main() 