import cv2
import numpy as np

def detect_panicles(image, threshold_value=200, min_area=30, max_area=150):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_panicles = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            cv2.drawContours(detected_panicles, [contour], -1, (255, 0, 0), 2)  # Draw red contour

    panicle_count = len([contour for contour in contours if min_area <= cv2.contourArea(contour) <= max_area])

    return detected_panicles, panicle_count
