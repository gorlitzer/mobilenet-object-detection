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

## Running on Raspberry Pi with picamera2

To run this application on a Raspberry Pi using the built-in camera module, you need to install the picamera2 library and configure your camera properly.

### Installing picamera2

1. **Update your system**:
   ```bash
   sudo apt update
   sudo apt upgrade -y
   ```

2. **Install picamera2 and dependencies**:
   ```bash
   # For newer Raspberry Pi OS versions (Bullseye and later)
   sudo apt install -y python3-picamera2 python3-libcamera python3-kms++ python3-opencv
   
   # For older Raspberry Pi OS versions
   sudo apt install -y python3-picamera2 python3-opencv libcap-dev
   ```

3. **Enable the camera in raspi-config**:
   ```bash
   sudo raspi-config
   ```
   Navigate to "Interface Options" > "Camera" and enable it.

4. **Reboot your Raspberry Pi**:
   ```bash
   sudo reboot
   ```

5. **Verify the installation**:
   ```bash
   python3 -c "import picamera2; print('picamera2 is installed')"
   ```

6. **Test your camera**:
   ```bash
   # For newer Raspberry Pi OS versions (Bullseye and later)
   libcamera-hello
   
   # For older Raspberry Pi OS versions
   vcgencmd get_camera
   ```

### Using a Virtual Environment with picamera2

If you're using a virtual environment, make sure to create it with system packages:

```bash
# Deactivate current venv if active
deactivate

# Create a new venv with system packages
python3 -m venv --system-site-packages venv
source venv/bin/activate

# Install your requirements
pip install -r requirements.txt
```

### Troubleshooting picamera2 Installation

If you encounter issues with picamera2, try these troubleshooting steps:

1. **Check if picamera2 is installed system-wide**:
   ```bash
   python3 -c "import picamera2; print('picamera2 is installed')"
   ```

2. **Check your Python path**:
   ```bash
   python3 -c "import sys; print('\n'.join(sys.path))"
   ```

3. **Find where picamera2 is installed**:
   ```bash
   find /usr -name "picamera2" -type d 2>/dev/null
   ```

4. **Check your Raspberry Pi OS version**:
   ```bash
   cat /etc/os-release
   ```

5. **Check if your camera is properly connected**:
   ```bash
   # For older Raspberry Pi OS versions
   vcgencmd get_camera
   
   # For newer Raspberry Pi OS versions (Bullseye and later)
   libcamera-hello
   ```
   This should show information about your camera if it's properly connected and enabled.

6. **If you see "vc_gencmd_read_response returned -1 error=1 error_msg='Command not registered'"**:
   This error occurs on newer Raspberry Pi OS versions (Bullseye and later) because they use the new libcamera stack instead of the legacy camera stack. Use `libcamera-hello` instead of `vcgencmd get_camera` to check your camera status.

### Alternative: Using OpenCV with a USB Camera

If you continue to have issues with picamera2, you can use OpenCV with a USB camera instead:

1. Edit your `config.py` file to set `USE_PICAMERA = False`
2. Connect a USB camera to your Raspberry Pi
3. Run your application with the USB camera as the source

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