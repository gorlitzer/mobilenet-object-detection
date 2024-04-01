#!/usr/bin/env python3

import cv2
from picamera2 import Picamera2
import numpy as np
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from visualize_detections import objectRecognition


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
        while True:
            pc2array = picam2.capture_array()
            result, _ = objectRecognition(dnn, classNames, pc2array, 0.6, 0.6)
            ret, buffer = cv2.imencode(".jpg", result)
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


(dnn, classNames) = configDNN()

picam2 = Picamera2()
config = picam2.configure(
    picam2.create_video_configuration(main={"size": (1920, 1080), "format": "RGB888"})
)
picam2.configure(config)
picam2.start()


def start_server():
    server_address = ("", 8000)
    httpd = HTTPServer(server_address, VideoStreamHandler)
    print("Server started on port 8000")
    httpd.serve_forever()


if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        picam2.stop()
