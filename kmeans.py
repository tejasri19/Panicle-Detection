import cv2
import numpy as np

def kmeans_panicle_detection(input_image):
    # Convert the image to the HSV color space
    hsv_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

    # Threshold value for the V channel (brightness)
    threshold_value = 170  # Adjust this value as needed

    # Create a binary mask based on the threshold
    binary_mask = (hsv_img[:, :, 2] > threshold_value).astype(np.uint8) * 255

    # Save the binary mask
    cv2.imwrite('binary_mask.jpg', binary_mask)

    # Load the binary image
    binary_image = cv2.imread("binary_mask.jpg", cv2.IMREAD_GRAYSCALE)

    # Threshold value (adjust as needed)
    threshold_value = 200

    # Apply a binary threshold to the image
    _, binary_thresholded = cv2.threshold(binary_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours in the binary thresholded image
    contours, _ = cv2.findContours(binary_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Minimum contour area threshold to filter out noise (adjust as needed)
    min_contour_area = 20

    # Create an empty mask to store the filtered binary image
    filtered_image = np.zeros_like(binary_image)

    # Iterate through the contours
    for idx, contour in enumerate(contours):
        contour_area = cv2.contourArea(contour)

        if contour_area >= min_contour_area:
            cv2.drawContours(filtered_image, [contour], -1, 255, thickness=cv2.FILLED)

    # Save the filtered image
    cv2.imwrite("filtered_image.png", filtered_image)

    # Load the filtered image (where only panicles are present)
    filtered_image = cv2.imread("filtered_image.png", cv2.IMREAD_GRAYSCALE)

    # Find contours in the filtered image
    contours, _ = cv2.findContours(filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the filtered image to draw rectangles on
    image_with_rectangles = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)

    # Iterate through the contours and draw rectangles on the image
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image_with_rectangles, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Replace rectangles on the original image with rectangles from image_with_rectangles
    result_image = input_image.copy()
    result_image[image_with_rectangles > 0] = image_with_rectangles[image_with_rectangles > 0]

    num_boxes = len(contours)
    num_panicles= num_boxes
    return result_image, num_panicles