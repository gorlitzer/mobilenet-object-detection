#!/bin/bash

# Create necessary directories
mkdir -p models data/detections

# Check if running on Raspberry Pi
if [ -f /proc/device-tree/model ]; then
    PI_MODEL=$(cat /proc/device-tree/model | tr -d '\0')
    if [[ $PI_MODEL == *"Raspberry Pi"* ]]; then
        echo "Raspberry Pi detected: $PI_MODEL"
        
        # Install system dependencies
        echo "Installing system dependencies..."
        sudo apt update
        sudo apt install -y python3-picamera2 python3-opencv libcap-dev python3-libcamera python3-kms++
        
        # Check if picamera2 is installed
        if ! command -v python3-picamera2 &> /dev/null; then
            echo "Installing picamera2..."
            sudo apt install -y python3-picamera2
        else
            echo "picamera2 is already installed"
        fi
        
        # Check if camera is enabled
        if ! vcgencmd get_camera 2>/dev/null | grep -q "supported=1"; then
            echo "WARNING: Camera may not be enabled. Please run 'sudo raspi-config' and enable the camera."
        fi
    fi
fi

# Download YOLOv8n model
echo "Downloading YOLOv8n model..."
wget -O models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

echo "Setup completed successfully!" 