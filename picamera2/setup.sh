#!/bin/bash

# Define the URL to download the zip file
url="https://core-electronics.com.au/media/kbase/491/Object_Detection_Files.zip"


# Download the zip file
wget -O "Object_Detection_Files.zip" "$url"

# Extract the contents of the zip file into the specified directory
unzip "Object_Detection_Files.zip" -d "$directory"

# Remove the zip file after extraction
rm "Object_Detection_Files.zip"

