import numpy as np
import cv2


def objectRecognition(dnn, classNames, image, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = dnn.detect(image, confThreshold=thres, nmsThreshold=nms)

    # Define a dictionary to map class labels to colors
    class_colors = {
        "person": (0, 255, 0),  # Green color for person
        "car": (255, 0, 0),  # Blue color for car
        "dog": (0, 0, 255),  # Red color for dog (example, you can add more)
        # Add more class labels and their corresponding colors as needed
    }

    if len(objects) == 0:
        objects = classNames
    recognisedObjects = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                recognisedObjects.append([box, className])
                if draw:
                    color = class_colors.get(
                        className, (0, 0, 255)
                    )  # Default color: red
                    cv2.rectangle(image, box, color=color, thickness=1)
                    cv2.putText(
                        image,
                        classNames[classId - 1]
                        + " ("
                        + str(round(confidence * 100, 2))
                        + ")",
                        (box[0] - 10, box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

    return image, recognisedObjects
