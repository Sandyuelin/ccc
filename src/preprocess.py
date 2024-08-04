import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import logging


def is_dark_background(img):
    """
    Check if the background is dark based on specific regions.
    """
    h, w = img.shape[:2]

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define regions
    regions = {
        'left': gray[:, :int(0.1 * w)],
        'right': gray[:, int(0.9 * w):],
        'top': gray[:int(0.1 * h), :],
        'bottom': gray[int(0.9 * h):, :],
        'center': gray[int(0.3 * h):int(0.7 * h), int(0.3 * w):int(0.7 * w)]
    }

    # Calculate means
    region_means = {key: np.mean(region) for key, region in regions.items()}

    # Check if center mean is greater than any of the edge means
    center_mean = region_means['center']
    edge_means = [region_means[key] for key in regions if key != 'center']
    # 去除edge_means最大值
    edge_means.remove(max(edge_means))

    return center_mean > max(edge_means)


def invert_image(img):
    """
    Invert the image colors.
    """
    img = 255 - img
    img[:, :20, :] = 255
    img[:, -20:, :] = 255
    img[:20, :, :] = 255
    img[-20:, :, :] = 255
    return img


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
    if len(max_ls) >= 10:
        if max_ls[2][0] >= max_ls[3][0] +20:
            max_ls = max_ls[2:-2]
        else:
            max_ls = max_ls[1:-1]
    # 补丁
    while True:
        right_max = max_ls[0]
        if right_max[0] < 60:
            max_ls = max_ls[1:]
        else:
            break
    # print(max_ls)
    left_max = None
    while True:
        max_ls = max_ls[1:]
        if len(max_ls) == 0:
            break
        left_max = max_ls[0]
        if left_max[0] < right_max[0] and left_max[0] < 120:
            break

    if not left_max or not left_max[0] < right_max[0]:
        return -1  # Return -1 for unsuccessful thresholding

    ss = left_max[1] + right_max[1]
    thresh = int(left_max[0] * left_max[1] / ss + right_max[0] * right_max[1] / ss)
    thresh = int(thresh * (256 / bin_n))
    if thresh < 80:
        thresh = int(thresh * (1 + (80 - thresh) / 30))
    return thresh


def process_one_image(img):
    """
    Process a single image using dynamic thresholding.
    """
    if img is None:
        raise ValueError("Image is None")
    dark = 0
    if is_dark_background(img):
        # print("dark")
        dark = 1
        img = invert_image(img)
        img = cv2.convertScaleAbs(img, alpha=0.98, beta=-10)
        # plt.imshow(img)
        # plt.show()

    if len(img.shape) == 3:  # Process color images
        channels = np.split(img, 3, axis=2)
        ch_list = []
        for id, channel in enumerate(channels):
            thresh = dynamic_thresh(channel)
            # print(thresh)
            if thresh != -1:
                condition = channel >= thresh
                bias = 0
                if thresh >= 80:

                    bias = (50 - thresh) * 2
                distances = np.where(condition, np.minimum((channel - thresh) / 2, 1.5), 0)
                result = np.where(condition, (thresh + (distances ** 14)).astype(int), channel + bias)
                channel = np.clip(result, 0, 255).astype(np.uint8)
                ch_list.append(channel)
            else:
                print(f"Error in dynamic_thresh for channel {id}. Returning original image.")
                return img

        img_p = np.concatenate(ch_list, axis=2)
        # Apply median blur to remove noise
        kernel_size = 5
        img_p = cv2.medianBlur(img_p, kernel_size)

        # dilate the image
        kernel = np.ones((3, 3), np.uint8)
        kernel2 = np.ones((5, 5), np.uint8)
        img_p = cv2.dilate(img_p, kernel, iterations=1 if not dark else 4)
        # erode
        img_p = cv2.erode(img_p, kernel2, iterations=1 if not dark else 0)
        # plt.imshow(img_p)
        # plt.show()

        # # 图像加对比度
        img_p = cv2.convertScaleAbs(img_p, alpha=3.5, beta=-15).astype(np.uint8)

        # 对图像中的像素，做判断：其三个通道有两个大于等于250，如果是，这个像素就是白色的
        # for i in range(img_p.shape[0]):
        #      for j in range(img_p.shape[1]):
        #          if (img_p[i, j, 0] >= 250 and img_p[i, j, 1] >= 250) or (img_p[i, j, 0] >= 250 and img_p[i, j, 2] >= 250) or (img_p[i, j, 1] >= 250 and img_p[i, j, 2] >= 250):
        #             img_p[i, j] = 255
        img_p = np.where(np.any(img_p >= 250, axis=2, keepdims=True), [255, 255, 255], img_p)

        return img_p

    else:
        raise ValueError("Image is not a color image with 3 channels")


def preprocess_images_from_directory(input_dir, output_dir):
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

                    # Calculate padding to make the image square
                    h, w, _ = preprocessed_image.shape
                    if h > w:
                        pad = (h - w) // 2
                        preprocessed_image = cv2.copyMakeBorder(preprocessed_image, 0, 0, pad, pad, cv2.BORDER_CONSTANT,
                                                                value=[255, 255, 255])
                    else:
                        pad = (w - h) // 2
                        preprocessed_image = cv2.copyMakeBorder(preprocessed_image, pad, pad, 0, 0, cv2.BORDER_CONSTANT,
                                                                value=[255, 255, 255])


                    # Convert to PIL Image
                    preprocessed_image_pil = Image.fromarray(preprocessed_image.astype(np.uint8))


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
    output_directory = r"C:\Users\dell\Desktop\preprocessed"  # Update with the actual path
    preprocess_images_from_directory(input_directory, output_directory)

    # in_dir = r"C:\Users\dell\Desktop\2.jpg"
    # raw_img = cv2.imread(in_dir)
    # porcessed = process_one_image(raw_img)
    # # plot processed
    #
    # plt.imshow(porcessed)
    # plt.show()
    #
