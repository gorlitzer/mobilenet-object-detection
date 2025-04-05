import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model configuration
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'yolov8n.pt')
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Road sign classes (COCO dataset classes that are relevant for road signs)
ROAD_SIGN_CLASSES = [
    'stop sign',
    'traffic light',
    'bench',  # Some traffic signs might be detected as benches
    'clock',  # Some circular signs might be detected as clocks
]

# Camera configuration
USE_PICAMERA = True  # Set to True for Raspberry Pi with picamera2
CAMERA_CONFIG = {
    'main': {
        'size': (640, 480),
        'format': 'RGB888',
        'buffer_count': 1,
    },
    'lores': {
        'size': (320, 240),
        'format': 'YUV420',
    },
    'display': 'lores',
    'encode': 'main',
}

# Video configuration (for non-picamera sources)
DEFAULT_CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Telegram configuration
TELEGRAM_ENABLED = os.getenv('TELEGRAM_ENABLED', 'false').lower() == 'true'
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
NOTIFICATION_COOLDOWN = int(os.getenv('NOTIFICATION_COOLDOWN', '60'))  # seconds

# Display configuration
DISPLAY_FPS = True
DISPLAY_CONFIDENCE = True
DISPLAY_BOUNDING_BOX = True
BOUNDING_BOX_COLOR = (0, 255, 0)  # BGR format
BOUNDING_BOX_THICKNESS = 2
TEXT_COLOR = (0, 255, 0)  # BGR format
TEXT_THICKNESS = 2
TEXT_SCALE = 0.6 