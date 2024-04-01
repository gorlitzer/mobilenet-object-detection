#!/usr/bin/env python3

# https://leoncode.co.uk/posts/object-detection-recognition-raspberry-pi-camera/
import cv2
from picamera2 import Picamera2
import numpy as np


def configDNN():
    classNames = []
    classFile = "Object_Detection_Files/coco.names"
    with open(classFile, "rt") as f:
        classNames = f.read().rstrip("\n").split("\n")

    # This is to pull the information about what each object should look like
    configPath = "Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    weightsPath = "Object_Detection_Files/frozen_inference_graph.pb"

    dnn = cv2.dnn_DetectionModel(weightsPath, configPath)
    dnn.setInputSize(320, 320)
    dnn.setInputScale(1.0 / 127.5)
    dnn.setInputMean((127.5, 127.5, 127.5))
    dnn.setInputSwapRB(True)

    return (dnn, classNames)


# thres = confidence threshold before an object is detected
# nms = Non-Maximum Suppression - higher percentage reduces number of overlapping detected boxes
# cup = list of names of objects to detect or empty for all available objects
def objectRecognition(dnn, classNames, image, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = dnn.detect(image, confThreshold=thres, nmsThreshold=nms)

    if len(objects) == 0:
        objects = classNames
    recognisedObjects = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                recognisedObjects.append([box, className])
                if draw:
                    cv2.rectangle(image, box, color=(0, 0, 255), thickness=1)
                    cv2.putText(
                        image,
                        classNames[classId - 1]
                        + " ("
                        + str(round(confidence * 100, 2))
                        + ")",
                        (box[0] - 10, box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )

    return image, recognisedObjects


# Below determines the size of the live feed window that will be displayed on the Raspberry Pi OS
if __name__ == "__main__":

    (dnn, classNames) = configDNN()

    picam2 = Picamera2()
    # Set the camera format to RGB instead of the default RGBA
    config = picam2.create_preview_configuration({"format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    while True:
        # Copy the camera image into an array
        pc2array = picam2.capture_array()

        # Rotate the image 180Degrees if the camera is upside down
        pc2array = np.rot90(pc2array, 2).copy()

        # Do the object recognition
        result, objectInfo = objectRecognition(dnn, classNames, pc2array, 0.6, 0.6)

        # Show it in a window
        cv2.imshow("Output", pc2array)
        cv2.waitKey(50)
