# Road Sign Detection Server

A simple web server that streams video from a Raspberry Pi camera with YOLO object detection for road signs.

## Features

- Real-time video streaming from Raspberry Pi camera
- YOLO object detection for road signs
- Web interface for viewing the video stream
- Configurable host and port
- Support for multiple road sign classes

## Requirements

- Python 3.8+
- Raspberry Pi with camera module
- OpenCV
- Ultralytics YOLO
- Picamera2

## Installation

2. Create a virtual environment and activate it:
```bash
python -m venv --system-site-packages venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download the YOLO model:
```bash
mkdir -p models
wget https://github.com/ultralytics/yolov8/releases/download/v0.0.0/yolov8n.pt -O models/yolov8n.pt
```

## Usage

Run the server with default settings (host: 0.0.0.0, port: 8000):
```bash
./main.py
```

Or specify custom host and port:
```bash
./main.py --host 127.0.0.1 --port 8080
```

Open your web browser and navigate to:
```
http://localhost:8000
```

## Configuration

You can modify the following settings in `main.py`:

- `CONFIDENCE_THRESHOLD`: Minimum confidence score for detections (default: 0.5)
- `ROAD_SIGN_CLASSES`: List of road sign classes to detect
- `CAMERA_CONFIG`: Camera configuration settings

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Picamera2](https://github.com/raspberrypi/picamera2)
- [OpenCV](https://opencv.org/)