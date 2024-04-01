# import libraries
import numpy as np
import cv2

from visualize_detections import visualize_detection

# 1. Initial requirements

# path to the prototxt file with text description of the network architecture
prototxt = "./models/MobileNetSSD/MobileNetSSD_deploy.prototxt"
# path to the .caffemodel file with learned network
caffe_model = "./models/MobileNetSSD/MobileNetSSD_deploy.caffemodel"

# read a network model (pre-trained) stored in Caffe framework's format
net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)

# dictionary with the object class id and names on which the model is trained
classNames = {
    0: "background",
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "pottedplant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tvmonitor",
}

# Define a color map for each class label for visualization
# Bunch of them don't work, don't know why. But at least the colors are different.
color_map = {
    "background": (0, 0, 0),  # Black for background
    "aeroplane": (255, 0, 0),  # Blue for aeroplane
    "bicycle": (0, 255, 0),  # Green for bicycle
    "bird": (0, 0, 255),  # Red for bird
    "boat": (255, 255, 0),  # Cyan for boat
    "bottle": (255, 0, 255),  # Magenta for bottle
    "bus": (0, 255, 255),  # Yellow for bus
    "car": (255, 255, 255),  # White for car
    "cat": (128, 128, 128),  # Gray for cat
    "chair": (128, 0, 0),  # Maroon for chair
    "cow": (128, 128, 0),  # Olive for cow
    "diningtable": (128, 0, 128),  # Purple for diningtable
    "dog": (0, 128, 128),  # Teal for dog
    "horse": (0, 128, 0),  # Green for horse
    "motorbike": (255, 165, 0),  # Orange (NOT WORKING) for motorbike
    "person": (0, 0, 128),  # Navy for person
    "pottedplant": (255, 69, 0),  # Orange-Red (NOT WORKING) for pottedplant
    "sheep": (255, 140, 0),  # Dark Orange for sheep
    "sofa": (255, 20, 147),  # Deep Pink for sofa
    "train": (0, 255, 127),  # Spring Green for train
    "tvmonitor": (75, 0, 130),  # Indigo for tvmonitor
}

# 2. Set up the camera

# capture the webcam feed
cap = cv2.VideoCapture(0)

while True:

    # 3. Format the input

    ret, frame = cap.read()

    # size of image
    width = frame.shape[1]
    height = frame.shape[0]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1 / 127.5,
        size=(300, 300),
        mean=(127.5, 127.5, 127.5),
        swapRB=True,
        crop=False,
    )

    # 4. Object detection and visualization

    # blob object is passed as input to the object
    net.setInput(blob)

    net.setInput(blob)
    detections = net.forward()

    # Visualize detections
    visualize_detection(frame, detections, classNames, color_map)

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) >= 0:  # Break with ESC
        break

cap.release()
cv2.destroyAllWindows()
