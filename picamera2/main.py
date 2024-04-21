#!/usr/bin/env python3

from datetime import datetime
import os
import cv2
from picamera2 import Picamera2
import numpy as np
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from visualize_detections import objectRecognition

import telegram
import asyncio


def send_screenshot(file_name):
    # Replace with your Telegram bot token
    bot_token = "XXXX:XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

    # Replace with the chat ID of the recipient
    chat_id = "XXXXXXXXX"

    # Create a Telegram Bot instance
    bot = telegram.Bot(token=bot_token)

    # Send the screenshot file
    with open(file_name, "rb") as file:
        bot.send_photo(chat_id=chat_id, photo=file)


def take_screenshot():
    try:
        # Create a folder with the current date if it doesn't exist
        folder_name = os.path.join("media", datetime.now().strftime("%Y-%m-%d"))
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Generate a file name with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(folder_name, f"screenshot_{timestamp}.jpg")

        # Take a screenshot
        pc2array = picam2.capture_array()
        cv2.imwrite(file_name, pc2array)

        return file_name
    except Exception as e:
        print(f"Error taking screenshot: {str(e)}")
        return ""


class VideoStreamHandler(BaseHTTPRequestHandler):
    async def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header(
                "Content-type", "multipart/x-mixed-replace; boundary=--frame"
            )
            self.end_headers()
            await self.stream()
        else:
            self.send_error(404)

    async def stream(self):
        while True:
            pc2array = picam2.capture_array()
            image, recognizedArray = objectRecognition(
                dnn, classNames, pc2array, 0.6, 0.6
            )

            person_detected = False
            for detected_object in recognizedArray:
                if detected_object[1] == "person":
                    person_detected = True
                    break

            # Take a screenshot and send it over Telegram when a person is detected
            if person_detected:
                file_name = take_screenshot()
                if file_name:
                    await send_screenshot(file_name)

            ret, buffer = cv2.imencode(".jpg", image)
            frame = buffer.tobytes()
            self.wfile.write(b"--frame\r\n")
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", len(frame))
            self.end_headers()
            self.wfile.write(frame)
            await asyncio.sleep(0.1)  # Adjust frame rate here


def configDNN():
    classNames = []
    classFile = "./Object_Detection_Files/coco.names"
    with open(classFile, "rt") as f:
        classNames = f.read().rstrip("\n").split("\n")

    configPath = "./Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    weightsPath = "./Object_Detection_Files/frozen_inference_graph.pb"

    dnn = cv2.dnn_DetectionModel(weightsPath, configPath)
    dnn.setInputSize(320, 320)
    dnn.setInputScale(1.0 / 127.5)
    dnn.setInputMean((127.5, 127.5, 127.5))
    dnn.setInputSwapRB(True)

    return (dnn, classNames)


def start_server():
    server_address = ("", 8000)
    httpd = HTTPServer(server_address, VideoStreamHandler)
    print("Server started on port 8000")
    httpd.serve_forever()


(dnn, classNames) = configDNN()

picam2 = Picamera2()
config = picam2.create_preview_configuration({"format": "RGB888", "size": (1920, 1080)})
picam2.configure(config)
picam2.start()


if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        picam2.stop()
