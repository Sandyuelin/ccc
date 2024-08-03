import cv2
import os
from PIL import Image
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert colors if necessary
    if np.sum(binary == 255) < np.sum(binary == 0):
        binary = cv2.bitwise_not(binary)

    # Convert back to BGR format
    bgr_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # Convert to PIL Image format
    pil_image = Image.fromarray(bgr_image)

    return pil_image

def preprocess_images_from_directory(input_dir, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all files and directories in the input directory
    input_path = Path(input_dir)
    for file_path in input_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            try:
                # Load image
                image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)

                if image is not None:
                    # Preprocess image
                    preprocessed_image = preprocess_image(image)

                    # Construct output path
                    relative_path = file_path.relative_to(input_path)
                    output_path = Path(output_dir) / relative_path

                    # Ensure the output directory exists
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    # Save the preprocessed image
                    preprocessed_image.save(str(output_path))
                else:
                    logging.warning(f"Skipping file {file_path} as it could not be loaded.")
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")

if __name__ == "__main__":
    input_directory = r"C:\Users\dell\Desktop\test_single_discriminate"  # Update with the actual path
    output_directory = r"C:\Users\dell\Desktop\preprocessed_images"  # Update with the actual path
    preprocess_images_from_directory(input_directory, output_directory)
