# Road Sign Detection using YOLO

This project implements real-time road sign detection using YOLO (You Only Look Once) object detection algorithm. It can detect and classify various types of road signs from video streams or images.

## Project structure 

```bash
yolo-signs/
├── src/
│   ├── main.py           # Main application entry point
│   ├── detector.py       # YOLO detector implementation
│   ├── utils.py          # Utility functions
│   └── config.py         # Configuration settings
├── models/               # Pre-trained YOLO models
├── data/                 # Sample images and videos
├── requirements.txt      # Project dependencies
├── .env.example         # Example environment variables
├── setup.sh             # Setup script
└── README.md            # Project documentation
```

## Features

- Real-time road sign detection using YOLO
- Support for both Raspberry Pi camera (picamera2) and standard webcams
- Configurable detection confidence threshold
- Support for multiple road sign classes
- Optional Telegram notifications for detected signs
- FPS counter and display
- Automatic saving of detected frames

## Prerequisites

- Python 3.8 or higher
- OpenCV
- PyTorch
- CUDA (optional, for GPU acceleration)
- picamera2 (for Raspberry Pi)

## Installation

### On Raspberry Pi

1. Install system dependencies:
```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

### On Other Platforms (Linux, macOS, Windows)

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

## Usage

1. Run the detection:

### On Raspberry Pi (using picamera2)
```bash
python src/main.py
```

### On Other Platforms (using webcam or video file)
```bash
# Using webcam
python src/main.py --source 0

# Using video file
python src/main.py --source path/to/video.mp4
```

Press 'q' to quit the application.

## Configuration

You can modify the following settings in `src/config.py`:

- `USE_PICAMERA`: Set to `True` to use picamera2 on Raspberry Pi
- `CONFIDENCE_THRESHOLD`: Minimum confidence for detections
- `ROAD_SIGN_CLASSES`: List of road sign classes to detect
- `CAMERA_CONFIG`: Configuration for picamera2

## Supported Road Signs

The model is trained to detect common road signs including:
- Stop signs
- Yield signs
- Speed limit signs
- Traffic lights
- Warning signs
- And more...

## Troubleshooting

### Raspberry Pi Camera Issues

If you encounter issues with the Raspberry Pi camera:

1. Make sure picamera2 is installed:
```bash
sudo apt install -y python3-picamera2
```

2. Check if your camera is enabled:
```bash
sudo raspi-config
# Navigate to Interface Options > Camera > Enable
```

3. Verify camera connection:
```bash
libcamera-hello
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 