import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

def apply_morphological_operations(image, dilation_kernel_size=(5, 5), erosion_kernel_size=(5, 5)):
    """
    Apply dilation and erosion to the image.
    """
    # Convert PIL image to numpy array (grayscale for operations)
    image_np = np.array(image.convert('L'))  # Convert to grayscale
    
    # Threshold to binary image
    _, binary_image = cv2.threshold(image_np, 1, 255, cv2.THRESH_BINARY)
    
    # Define kernels for dilation and erosion
    dilation_kernel = np.ones(dilation_kernel_size, np.uint8)
    erosion_kernel = np.ones(erosion_kernel_size, np.uint8)
    
    # Apply dilation
    dilated_image = cv2.dilate(binary_image, dilation_kernel, iterations=1)
    
    # Apply erosion
    eroded_image = cv2.erode(dilated_image, erosion_kernel, iterations=1)
    
    # Convert back to PIL image
    final_image = Image.fromarray(eroded_image, 'L').convert('RGB')
    
    return final_image

def create_calligraphy_image(text, output_path, image_size=(224, 224), font_size=150):
    """
    Create an image with Chinese calligraphy text, apply bold effect, and morphological operations.
    """
    # Create a new image with white background
    image = Image.new('RGB', image_size, 'white')
    draw = ImageDraw.Draw(image)
    
    # Load the SimSun font (adjust the path to your SimSun font file)
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\simsun.ttc", font_size)
    except IOError:
        print("Font file not found. Please ensure the font path is correct.")
        return
    
    # Calculate text size and position
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    text_x = (image_size[0] - text_width) // 2
    text_y = (image_size[1] - text_height) // 2
    
    # Draw the text on the image with an outline to simulate boldness
    for offset in range(-5, 6):  # Increase offset range for bold effect
        draw.text((text_x + offset, text_y), text, fill='black', font=font)
        draw.text((text_x, text_y + offset), text, fill='black', font=font)
    
    # Draw the main text on top
    draw.text((text_x, text_y), text, fill='black', font=font)
    
    # Apply dilation and erosion effect
    image_with_morphology = apply_morphological_operations(image)
    
    # Save the image
    image_with_morphology.save(output_path)
    print(f"Image saved to {output_path}")

def process_json(json_file, output_dir, image_size=(224, 224), font_size=150):
    """
    Process JSON file and create calligraphy images based on the texts.
    """
    # Load JSON data
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each character
    for key, characters in data.items():
        print(f"Processing character: {key}")
        for text in characters:
            # Ensure text is valid for filename (remove invalid characters)
            safe_text = text.translate(str.maketrans('', '', '<>:\"/\\|?*')).replace(' ', '_')
            
            # Construct output path
            output_path = os.path.join(output_dir, f"{safe_text}.png")
            
            # Create the calligraphy image
            create_calligraphy_image(text, output_path, image_size, font_size)

if __name__ == "__main__":
    json_file = r"C:\Users\dell\Desktop\chinese_calligraphy_classifier\simplified_complex.json"  # Update with your actual JSON file path
    output_directory = r"C:\Users\dell\Desktop\calligraphy_images"  # Update with your actual output directory
    process_json(json_file, output_directory)
