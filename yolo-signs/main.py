#!/usr/bin/env python3

"""
Road Sign Detection Server
A simple web server that streams video from a Raspberry Pi camera with YOLO object detection.
"""

import cv2
import numpy as np
import threading
import time
import logging
import argparse
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
from ultralytics import YOLO
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIDENCE_THRESHOLD = 0.5
ROAD_SIGN_CLASSES = ['stop sign', 'traffic light', 'bench', 'clock']
CAMERA_CONFIG = {
    'main': {
        'format': 'RGB888',
        'size': (640, 480)
    }
}

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
    </style>
</head>
<body>
    <div class="container">
        <h1>Road Sign Detection</h1>
        <div class="video-container">
            <img src="stream.mjpg" alt="Video Stream">
        </div>
    </div>
</body>
</html>
"""

class StreamingOutput(io.BufferedIOBase):
    """Class to handle streaming output."""
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()
        self._buffer = io.BytesIO()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()
        return len(buf)

    def read(self, size=-1):
        return self._buffer.read(size)

    def readable(self):
        return True

    def writable(self):
        return True

    def seekable(self):
        return False

class StreamingHandler(BaseHTTPRequestHandler):
    """HTTP request handler for streaming video."""
    
    def do_GET(self):
        if self.path == "/":
            self.send_response(301)
            self.send_header("Location", "/index.html")
            self.end_headers()
        elif self.path == "/index.html":
            content = HTML_TEMPLATE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", 0)
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=FRAME"
            )
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except Exception as e:
                logging.warning(
                    "Removed streaming client %s: %s", self.client_address, str(e)
                )
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, HTTPServer):
    """Threading HTTP server for streaming."""
    allow_reuse_address = True
    daemon_threads = True

def detect_objects(frame, model):
    """Detect objects in the frame using YOLO."""
    # Run YOLO detection
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    
    # Process detections
    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = r
        class_name = results.names[int(class_id)]
        
        # Only process road sign related classes
        if class_name in ROAD_SIGN_CLASSES:
            # Draw bounding box
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )
            
            # Prepare label text
            label = f"{class_name} {score:.2f}"
            
            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )
            cv2.rectangle(
                frame,
                (int(x1), int(y1) - label_height - 10),
                (int(x1) + label_width, int(y1)),
                (0, 255, 0),
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
    
    return frame

def process_frames():
    """Process frames from the camera and stream them."""
    global picam2, model, output
    
    # Initialize camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main=CAMERA_CONFIG['main'])
    picam2.configure(config)
    picam2.start()
    logger.info("Camera started successfully")
    
    # Initialize YOLO model
    model_path = os.path.join('models', 'yolov8n.pt')
    model = YOLO(model_path)
    logger.info(f"Loaded YOLO model from {model_path}")
    
    # Initialize streaming output
    output = StreamingOutput()
    
    # Start recording
    picam2.start_recording(JpegEncoder(), FileOutput(output))
    logger.info("Started recording")
    
    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()
            
            # Convert from RGB to BGR (OpenCV format)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Detect objects
            frame = detect_objects(frame, model)
            
            # Convert back to RGB for JPEG encoding
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ret:
                continue
            
            # Update streaming output
            with output.condition:
                output.frame = buffer.tobytes()
                output.condition.notify_all()
            
            # Sleep to control frame rate
            time.sleep(0.01)
    
    except Exception as e:
        logger.error(f"Error processing frames: {e}")
    finally:
        picam2.stop_recording()
        picam2.stop()
        logger.info("Camera stopped")

def start_server(host='0.0.0.0', port=8000):
    """Start the streaming server."""
    try:
        address = (host, port)
        server = StreamingServer(address, StreamingHandler)
        logger.info(f"Server started on {host}:{port}")
        server.serve_forever()
    except Exception as e:
        logger.error(f"Error starting server: {e}")

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='Road Sign Detection Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    args = parser.parse_args()
    
    # Start processing thread
    process_thread = threading.Thread(target=process_frames)
    process_thread.daemon = True
    process_thread.start()
    
    # Start server
    start_server(args.host, args.port)

if __name__ == "__main__":
    main() 