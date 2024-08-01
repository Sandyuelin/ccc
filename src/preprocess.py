import cv2
import numpy as np
from PIL import Image


def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Invert colors if necessary
    if np.sum(binary == 255) > np.sum(binary == 0):
        binary = cv2.bitwise_not(binary)

    # Convert back to BGR format
    bgr_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # Convert to PIL Image format
    pil_image = Image.fromarray(bgr_image)

    return pil_image
