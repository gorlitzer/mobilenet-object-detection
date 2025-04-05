#!/bin/bash

# Script to install and configure picamera2 on Raspberry Pi
echo "Starting picamera2 installation and configuration..."

# Check if running on Raspberry Pi
if [ ! -f /proc/device-tree/model ]; then
    echo "This script should be run on a Raspberry Pi."
    exit 1
fi

PI_MODEL=$(cat /proc/device-tree/model | tr -d '\0')
if [[ ! $PI_MODEL == *"Raspberry Pi"* ]]; then
    echo "This script should be run on a Raspberry Pi."
    exit 1
fi

echo "Raspberry Pi detected: $PI_MODEL"

# Update package lists
echo "Updating package lists..."
sudo apt update

# Install picamera2 and dependencies
echo "Installing picamera2 and dependencies..."
sudo apt install -y python3-picamera2 python3-opencv libcap-dev python3-libcamera python3-kms++

# Verify picamera2 installation
if ! python3 -c "import picamera2" 2>/dev/null; then
    echo "ERROR: picamera2 module not found after installation."
    echo "Trying alternative installation method..."
    
    # Try installing via pip
    sudo apt install -y python3-pip
    sudo pip3 install picamera2
    
    # Check again
    if ! python3 -c "import picamera2" 2>/dev/null; then
        echo "ERROR: Failed to install picamera2. Please try manually:"
        echo "sudo apt install -y python3-picamera2"
        exit 1
    fi
fi

echo "picamera2 module successfully installed!"

# Check if camera is enabled
echo "Checking camera status..."
if ! vcgencmd get_camera 2>/dev/null | grep -q "supported=1"; then
    echo "WARNING: Camera may not be enabled."
    echo "Please run 'sudo raspi-config' and enable the camera in Interface Options."
    echo "After enabling, reboot your Raspberry Pi."
fi

# Test camera access
echo "Testing camera access..."
if python3 -c "
from picamera2 import Picamera2
picam2 = Picamera2()
picam2.close()
" 2>/dev/null; then
    echo "Camera access test successful!"
else
    echo "WARNING: Could not access camera. Please check your camera connection and enable it in raspi-config."
fi

echo "Installation completed. If you encountered any warnings, please address them before running the application."
echo "You may need to reboot your Raspberry Pi for changes to take effect." 