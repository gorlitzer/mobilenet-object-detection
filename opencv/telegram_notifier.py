import os
import time
import cv2
from datetime import datetime
import telebot
from threading import Lock
from dotenv import load_dotenv

load_dotenv()

class TelegramNotifier:
    def __init__(self, bot_token=None, chat_id=None, cooldown_seconds=None):
        self.bot = telebot.TeleBot(
            bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        )
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.cooldown_seconds = cooldown_seconds or int(os.getenv('NOTIFICATION_COOLDOWN', '60'))
        self.last_notification_time = 0
        self.lock = Lock()
        
    def _can_send_notification(self):
        current_time = time.time()
        return (current_time - self.last_notification_time) >= self.cooldown_seconds
        
    def capture_video(self, frame_source, duration=10, fps=20):
        frames = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            frame = frame_source()
            if frame is not None:
                frames.append(frame)
            time.sleep(1/fps)
            
        if frames:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = f"detection_{timestamp}.mp4"
            
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            for frame in frames:
                out.write(frame)
            out.release()
            
            return video_path
        return None

    def notify_detection(self, frame_source=None):
        with self.lock:
            if not self._can_send_notification():
                return False
                
            try:
                # Update last notification time before attempting to send
                self.last_notification_time = time.time()
                
                # Send initial alert
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                message = f"⚠️ Person detected at {timestamp}"
                self.bot.send_message(self.chat_id, message)
                
                # Capture and send video if frame source is provided
                if frame_source:
                    video_path = self.capture_video(frame_source)
                    if video_path:
                        with open(video_path, 'rb') as video:
                            self.bot.send_video(self.chat_id, video)
                        os.remove(video_path)
                return True
                
            except Exception as e:
                print(f"Failed to send notification: {e}")
                # Reset the cooldown if sending fails
                self.last_notification_time = 0
                return False