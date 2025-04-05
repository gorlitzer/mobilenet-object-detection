import cv2
import numpy as np
import threading
import time
import logging
from flask import Flask, Response, render_template_string
from detector import RoadSignDetector
from utils import FPSCounter, draw_fps, save_frame
from config import USE_PICAMERA, CAMERA_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables
detector = RoadSignDetector()
fps_counter = FPSCounter()
frame_buffer = None
frame_lock = threading.Lock()
is_running = True

# HTML template for the web page
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Road Sign Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .video-container {
            text-align: center;
            margin: 20px 0;
        }
        img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .stats {
            margin-top: 20px;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 4px;
        }
        .detection-log {
            margin-top: 20px;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 4px;
            max-height: 200px;
            overflow-y: auto;
        }
        .detection-item {
            padding: 5px;
            border-bottom: 1px solid #eee;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Road Sign Detection</h1>
        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" alt="Video Stream">
        </div>
        <div class="stats">
            <p>FPS: <span id="fps">0</span></p>
        </div>
        <div class="detection-log">
            <h3>Recent Detections</h3>
            <div id="detections">
                <!-- Detections will be added here dynamically -->
            </div>
        </div>
    </div>
    <script>
        // Update FPS every second
        setInterval(function() {
            fetch('/fps')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                });
        }, 1000);
        
        // Update detections every 2 seconds
        setInterval(function() {
            fetch('/detections')
                .then(response => response.json())
                .then(data => {
                    const detectionsDiv = document.getElementById('detections');
                    detectionsDiv.innerHTML = '';
                    
                    data.detections.forEach(detection => {
                        const div = document.createElement('div');
                        div.className = 'detection-item';
                        div.textContent = `${detection.class} (${detection.confidence.toFixed(2)}) - ${detection.timestamp}`;
                        detectionsDiv.appendChild(div);
                    });
                });
        }, 2000);
    </script>
</body>
</html>
"""

# Store recent detections
recent_detections = []
MAX_DETECTIONS = 10

def add_detection(detection):
    """Add a detection to the recent detections list."""
    timestamp = time.strftime("%H:%M:%S")
    recent_detections.append({
        'class': detection['class'],
        'confidence': detection['confidence'],
        'timestamp': timestamp
    })
    
    # Keep only the most recent detections
    if len(recent_detections) > MAX_DETECTIONS:
        recent_detections.pop(0)

def camera_thread():
    """Thread function to capture and process video frames."""
    global frame_buffer, is_running
    
    # Initialize camera
    if USE_PICAMERA:
        try:
            from picamera2 import Picamera2
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(main=CAMERA_CONFIG['main'])
            picam2.configure(config)
            picam2.start()
            logger.info("Picamera2 started successfully")
        except ImportError:
            logger.error("picamera2 not available. Please install it for Raspberry Pi support.")
            is_running = False
            return
        except Exception as e:
            logger.error(f"Error initializing picamera2: {e}")
            is_running = False
            return
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Failed to open video source")
            is_running = False
            return
    
    try:
        while is_running:
            # Capture frame
            if USE_PICAMERA:
                frame = picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    continue
            
            # Process frame
            processed_frame, detections = detector.detect(frame)
            
            # Update FPS counter
            fps = fps_counter.update()
            processed_frame = draw_fps(processed_frame, fps)
            
            # Save detections
            if detections:
                save_frame(processed_frame, detections[0])
                add_detection(detections[0])
            
            # Update frame buffer
            with frame_lock:
                frame_buffer = processed_frame
            
            # Sleep to control frame rate
            time.sleep(0.01)
    
    finally:
        if USE_PICAMERA:
            picam2.stop()
        else:
            cap.release()
        logger.info("Camera thread stopped")

def generate_frames():
    """Generator function to yield video frames."""
    while is_running:
        with frame_lock:
            if frame_buffer is None:
                time.sleep(0.1)
                continue
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame_buffer)
            if not ret:
                continue
            
            # Yield frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        # Sleep to control frame rate
        time.sleep(0.01)

@app.route('/')
def index():
    """Render the main page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/fps')
def get_fps():
    """Return current FPS."""
    return {'fps': fps_counter.fps}

@app.route('/detections')
def get_detections():
    """Return recent detections."""
    return {'detections': recent_detections}

def start_server(host='0.0.0.0', port=5000):
    """Start the web server."""
    global is_running
    
    # Start camera thread
    camera_thread_obj = threading.Thread(target=camera_thread)
    camera_thread_obj.daemon = True
    camera_thread_obj.start()
    
    # Start Flask server
    app.run(host=host, port=port, threaded=True)

if __name__ == '__main__':
    start_server() 