import numpy as np
import cv2


def visualize_detection(frame, detections, classNames, color_map):
    detected_objects = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.75:
            class_id = int(detections[0, 0, i, 1])
            
            # Store detection info
            detected_objects.append({
                "class": classNames[class_id],
                "confidence": confidence,
                "bbox": {
                    "x1": int(detections[0, 0, i, 3] * frame.shape[1]),
                    "y1": int(detections[0, 0, i, 4] * frame.shape[0]),
                    "x2": int(detections[0, 0, i, 5] * frame.shape[1]),
                    "y2": int(detections[0, 0, i, 6] * frame.shape[0])
                }
            })
            
            # Draw bounding box
            x_top_left = int(detections[0, 0, i, 3] * frame.shape[1])
            y_top_left = int(detections[0, 0, i, 4] * frame.shape[0])
            x_bottom_right = int(detections[0, 0, i, 5] * frame.shape[1])
            y_bottom_right = int(detections[0, 0, i, 6] * frame.shape[0])
            
            color = color_map.get(classNames[class_id], (0, 0, 0))
            
            cv2.rectangle(
                frame,
                (x_top_left, y_top_left),
                (x_bottom_right, y_bottom_right),
                color,
                thickness=2,
            )
            
            # Create and draw label
            label_val = "%.3f" % confidence
            label = classNames[class_id] + ": " + str(label_val)
            (w, h), t = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            
            cv2.rectangle(
                frame,
                (x_top_left, y_top_left - h),
                (x_top_left + w, y_top_left + t),
                color,
                cv2.FILLED,
            )
            
            cv2.putText(
                frame,
                label,
                (x_top_left, y_top_left),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                thickness=2,
            )
    
    return detected_objects