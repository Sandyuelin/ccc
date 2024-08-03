from PIL import Image, ImageOps
import cv2
import numpy as np
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def preprocess_image(image, target_size):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert colors if necessary
    if np.sum(binary == 255) < np.sum(binary == 0):
        binary = cv2.bitwise_not(binary)

    # Convert to PIL Image format
    pil_image = Image.fromarray(binary)

    # Resize the image while maintaining aspect ratio
    pil_image = ImageOps.contain(pil_image, (target_size, target_size), method=Image.Resampling.LANCZOS)

    # Create a new white background image
    new_image = Image.new("L", (target_size, target_size), 255)
    new_image.paste(pil_image, ((target_size - pil_image.width) // 2, (target_size - pil_image.height) // 2))

    # Convert to RGB (optional, if you need RGB format)
    new_image = new_image.convert("RGB")

    return new_image

def preprocess_images_from_directory(input_dir, output_dir, target_size):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all files and directories in the input directory
    input_path = Path(input_dir)
    for file_path in input_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            try:
                # Load image using OpenCV with Unicode path handling
                image = cv2.imdecode(np.fromfile(str(file_path), dtype=np.uint8), cv2.IMREAD_COLOR)

                if image is not None:
                    # Preprocess image
                    preprocessed_image = preprocess_image(image, target_size)

                    # Construct output path
                    relative_path = file_path.relative_to(input_path)
                    output_path = Path(output_dir) / relative_path

                    # Ensure the output directory exists
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    # Save the preprocessed image using PIL
                    preprocessed_image.save(str(output_path))
                else:
                    logging.warning(f"Skipping file {file_path} as it could not be loaded.")
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")

if __name__ == "__main__":
    input_directory = r"C:\Users\dell\Desktop\test_single_discriminate"  # Update with the actual path
    output_directory = r"C:\Users\dell\Desktop\preprocessed"  # Update with the actual path
    target_size = 256  # Desired size for the larger dimension
    preprocess_images_from_directory(input_directory, output_directory, target_size)
