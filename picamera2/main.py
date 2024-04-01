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


def record_video():
    # Create a folder with the current date if it doesn't exist
    folder_name = datetime.now().strftime("%Y-%m-%d")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Generate a file name with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = os.path.join(folder_name, f"video_{timestamp}.avi")

    # Start recording
    out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*"XVID"), 20.0, (640, 480))

    return out, file_name


def stop_recording(out):
    # Release the video writer
    out.release()


class VideoStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header(
                "Content-type", "multipart/x-mixed-replace; boundary=--frame"
            )
            self.end_headers()
            self.stream()
        else:
            self.send_error(404)

    def stream(self):
        recording = False
        out = None
        cooldown_time = 5  # Recording cooldown time (seconds)
        last_detection_time = time.time()

        while True:
            pc2array = picam2.capture_array()
            image, recognizedArray = objectRecognition(
                dnn, classNames, pc2array, 0.6, 0.6
            )
            print(recognizedArray)
            # Check if a person is detected
            # if "person" in result:
            #     if not recording:
            #         # Start recording if not already recording
            #         recording = True
            #         out, file_name = record_video()
            #         print(f"Recording started: {file_name}")

            #     # Write the frame to the video
            #     out.write(result)
            #     last_detection_time = time.time()

            # elif recording and time.time() - last_detection_time > cooldown_time:
            #     # Stop recording if a person is no longer detected and cooldown time has elapsed
            #     recording = False
            #     stop_recording(out)
            #     print(f"Recording stopped: {file_name}")

            ret, buffer = cv2.imencode(".jpg", image)
            frame = buffer.tobytes()
            self.wfile.write(b"--frame\r\n")
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", len(frame))
            self.end_headers()
            self.wfile.write(frame)
            time.sleep(0.1)  # Adjust frame rate here


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
