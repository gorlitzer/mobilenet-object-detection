import cv2
import time
from typing import Optional
import logging
import os
from datetime import datetime
import telegram
from config import (
    TELEGRAM_ENABLED,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    NOTIFICATION_COOLDOWN,
    DISPLAY_FPS,
)

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

class TelegramNotifier:
    """Handle Telegram notifications for detected road signs."""
    
    def __init__(self):
        self.last_notification_time = 0
        self.bot = None
        
        if TELEGRAM_ENABLED:
            try:
                self.bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
                logger.info("Telegram bot initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Telegram bot: {e}")
                TELEGRAM_ENABLED = False
    
    def send_notification(self, detection: dict) -> None:
        """
        Send notification about detected road sign.
        
        Args:
            detection: Dictionary containing detection information
        """
        if not TELEGRAM_ENABLED or not self.bot:
            return
            
        current_time = time.time()
        if current_time - self.last_notification_time < NOTIFICATION_COOLDOWN:
            return
            
        try:
            message = (
                f"🚨 Road Sign Detected!\n"
                f"Type: {detection['class']}\n"
                f"Confidence: {detection['confidence']:.2f}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message
            )
            
            self.last_notification_time = current_time
            logger.info(f"Sent Telegram notification for {detection['class']}")
            
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")

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