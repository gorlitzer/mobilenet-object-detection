#!/bin/bash

# Create models directory if it doesn't exist
mkdir -p Object_Detection_Files

# Download the Object Detection Files
wget https://core-electronics.com.au/media/kbase/491/Object_Detection_Files.zip

# Unzip the files
unzip Object_Detection_Files.zip -d Object_Detection_Files/

# Clean up
rm Object_Detection_Files.zip

echo "Setup completed successfully!"