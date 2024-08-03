import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import logging


def dynamic_thresh(img):
    """
    Calculate the dynamic threshold for binarizing an image channel.
    Assumes the background is light and text is dark.
    """
    bin_n = 256  # Number of bins for histogram
    img = img[::2, ::2]  # Downsample image for speed
    hist = cv2.calcHist([img], [0], None, [bin_n], [0, 256]).reshape(-1)
    hist = np.log10(hist + 1)  # Use log to handle differences better

    # Find local maxima in the histogram
    max_ls = []
    if hist[0] > hist[1]:
        max_ls.append((0, hist[0]))
    for i in range(1, bin_n - 1):
        if (hist[i] - hist[i - 1]) * (hist[i + 1] - hist[i]) <= 0 and not (
                hist[i - 1] == hist[i] and hist[i] == hist[i + 1]):
            max_ls.append((i, hist[i]))
    if hist[-1] > hist[-2]:
        max_ls.append((bin_n - 1, hist[-1]))

    # Sort maxima by intensity
    max_ls = sorted(max_ls, key=lambda x: x[1], reverse=True)
    right_max = max_ls[0]

    left_max = None
    while True:
        max_ls = max_ls[1:]
        if len(max_ls) == 0:
            break
        left_max = max_ls[0]
        if left_max[0] < right_max[0] and left_max[0] < 80:
            break

    if not left_max or not left_max[0] < right_max[0]:
        return -1  # Return -1 for unsuccessful thresholding

    ss = left_max[1] + right_max[1]
    thresh = int(left_max[0] * left_max[1] / ss + right_max[0] * right_max[1] / ss)
    thresh = int(thresh * (256 / bin_n))

    return thresh


def process_one_image(img):
    """
    Process a single image using dynamic thresholding.
    """
    if img is None:
        raise ValueError("Image is None")

    if len(img.shape) == 3:  # Process color images
        channels = np.split(img, 3, axis=2)
        ch_list = []
        for id, channel in enumerate(channels):
            thresh = dynamic_thresh(channel)
            if thresh != -1:
                condition = channel >= thresh
                distances = np.where(condition, np.minimum((channel - thresh) / 2, 1.5), 0)
                result = np.where(condition, (thresh + (distances ** 14)).astype(int), channel)
                channel = np.clip(result, 0, 255).astype(np.uint8)
                ch_list.append(channel)
            else:
                print(f"Error in dynamic_thresh for channel {id}. Returning original image.")
                return img

        img_p = np.concatenate(ch_list, axis=2)

        # Apply median blur to remove noise
        kernel_size = 3
        img_p = cv2.medianBlur(img_p, kernel_size)

        return img_p

    else:
        raise ValueError("Image is not a color image with 3 channels")


def preprocess_images_from_directory(input_dir, output_dir, target_size):
    """
    Preprocess images from the input directory and save to the output directory.
    """
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
                    preprocessed_image = process_one_image(image)

                    # Resize image to target size
                    preprocessed_image = cv2.resize(preprocessed_image, (target_size, target_size),
                                                    interpolation=cv2.INTER_AREA)

                    # Convert to PIL Image
                    preprocessed_image_pil = Image.fromarray(preprocessed_image)

                    # Construct output path
                    relative_path = file_path.relative_to(input_path)
                    output_path = Path(output_dir) / relative_path

                    # Ensure the output directory exists
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    # Save the preprocessed image using PIL
                    preprocessed_image_pil.save(str(output_path))
                else:
                    logging.warning(f"Skipping file {file_path} as it could not be loaded.")
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")


if __name__ == "__main__":
    input_directory = r"C:\Users\dell\Desktop\test_single_discriminate"  # Update with the actual path
    output_directory = r"C:\Users\dell\Desktop\preprocessed_images"  # Update with the actual path
    target_size = 256  # Desired size for the larger dimension
    preprocess_images_from_directory(input_directory, output_directory, target_size)
