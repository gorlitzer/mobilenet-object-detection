#!/bin/bash

# Download MobileNetSSD Model definition
if [ ! -f "MobileNetSSD_deploy.prototxt" ]; then
    echo "Downloading MobileNetSSD model file definition..."
    url="https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.prototxt"
    wget -O ./opencv/models/MobileNetSSD/MobileNetSSD_deploy.prototxt "$url"
fi

# Download MobileNetSSD Pre-trained model weights
if [ ! -f "MobileNetSSD_deploy.caffemodel" ]; then
    echo "Downloading MobileNetSSD pre-trained model weights..."
    url="https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.caffemodel"
    wget -O ./opencv/models/MobileNetSSD/MobileNetSSD_deploy.caffemodel "$url"
fi


