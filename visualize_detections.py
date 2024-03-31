import numpy as np
import cv2


def visualize_detection(frame, detections, classNames, color_map):
    for i in range(detections.shape[2]):
        # Confidence of prediction
        confidence = detections[0, 0, i, 2]

        # Set confidence level threshold to filter weak predictions
        if confidence > 0.75:
            # Get class id
            class_id = int(detections[0, 0, i, 1])

            # Scale to the frame
            width = frame.shape[1]
            height = frame.shape[0]
            x_top_left = int(detections[0, 0, i, 3] * width)
            y_top_left = int(detections[0, 0, i, 4] * height)
            x_bottom_right = int(detections[0, 0, i, 5] * width)
            y_bottom_right = int(detections[0, 0, i, 6] * height)

            # Draw bounding box around the detected object with the corresponding color
            color = color_map.get(
                classNames[class_id], (0, 0, 0)
            )  # Default to black if class color not found

            cv2.rectangle(
                frame,
                (x_top_left, y_top_left),
                (x_bottom_right, y_bottom_right),
                color,
                thickness=2,
            )

            # Create label
            label_val = "%.3f" % (confidence)  # Round off to 3 decimal places
            label = classNames[class_id] + ": " + str(label_val)

            # Get width and height of the label string
            (w, h), t = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

            # Draw bounding box around the text
            cv2.rectangle(
                frame,
                (x_top_left, y_top_left - h),
                (x_top_left + w, y_top_left + t),
                color,
                cv2.FILLED,
            )

            # Draw label text
            cv2.putText(
                frame,
                label,
                (x_top_left, y_top_left),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                thickness=2,
            )
