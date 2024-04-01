#!/bin/bash

# Define the URL to download the zip file
url="https://core-electronics.com.au/media/kbase/491/Object_Detection_Files.zip"

# Define the directory where you want to extract the contents
directory="./picamera2"

# Create the directory if it doesn't exist
mkdir -p "$directory"

# Download the zip file
wget -O "$directory/Object_Detection_Files.zip" "$url"

# Extract the contents of the zip file into the specified directory
unzip "$directory/Object_Detection_Files.zip" -d "$directory"

# Remove the zip file after extraction
rm "$directory/Object_Detection_Files.zip"

