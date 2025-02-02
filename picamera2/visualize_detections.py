import numpy as np
import cv2


def objectRecognition(dnn, classNames, image, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = dnn.detect(image, confThreshold=thres, nmsThreshold=nms)

    # Define a dictionary to map class labels to colors
    class_colors = {
        "person": (0, 255, 0),  # Green color for person
        "car": (255, 0, 0),  # Blue color for car
        "dog": (0, 0, 255),  # Red color for dog
        "cat": (255, 255, 0),  # Cyan for cat
        "bird": (255, 0, 255),  # Magenta for bird
    }

    if len(objects) == 0:
        objects = classNames
        
    detected_objects = []
    
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                # Store detection info in consistent format
                detected_objects.append({
                    "class": className,
                    "confidence": confidence,
                    "bbox": {
                        "x1": int(box[0]),
                        "y1": int(box[1]),
                        "x2": int(box[0] + box[2]),
                        "y2": int(box[1] + box[3])
                    }
                })
                
                if draw:
                    color = class_colors.get(className, (0, 0, 255))  # Default: red
                    
                    # Draw bounding box
                    cv2.rectangle(
                        image,
                        (int(box[0]), int(box[1])),
                        (int(box[0] + box[2]), int(box[1] + box[3])),
                        color,
                        thickness=2
                    )
                    
                    # Create and draw label
                    label = f"{className} ({confidence * 100:.2f}%)"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    
                    # Draw label background
                    cv2.rectangle(
                        image,
                        (int(box[0]), int(box[1] - h - 5)),
                        (int(box[0] + w), int(box[1])),
                        color,
                        cv2.FILLED
                    )
                    
                    # Draw label text
                    cv2.putText(
                        image,
                        label,
                        (int(box[0]), int(box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2
                    )

    return image, detected_objects