#!/bin/bash

# Exit on error
set -e

echo "Setting up Road Sign Detection Server..."

# Check if running on Raspberry Pi
if [ ! -f /etc/rpi-issue ]; then
    echo "Warning: This script is designed for Raspberry Pi. Some features may not work on other systems."
fi

# Install system dependencies
if [ -f /etc/debian_version ]; then
    echo "Installing system dependencies..."
    sudo apt update
    sudo apt install -y python3-picamera2 python3-opencv python3-venv
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create models directory and download YOLO model
echo "Downloading YOLO model..."
mkdir -p models
if [ ! -f models/yolov8n.pt ]; then
    wget https://github.com/ultralytics/yolov8/releases/download/v0.0.0/yolov8n.pt -O models/yolov8n.pt
fi

# Make main.py executable
echo "Making main.py executable..."
chmod +x main.py

echo "Setup complete! You can now run the server with:"
echo "./main.py"
echo ""
echo "Or with custom host and port:"
echo "./main.py --host 127.0.0.1 --port 8080" 