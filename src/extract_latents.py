import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from pathlib import Path

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """
    Load and preprocess an image for VGG16.
    """
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def get_feature_maps(model, img_array):
    """
    Get feature maps for an image using a specific layer of the model.
    """
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('block5_conv3').output)
    feature_maps = intermediate_layer_model.predict(img_array)
    return feature_maps

def downsample_feature_maps(feature_maps, target_dim=512):
    """
    Downsample feature maps to the target dimension.
    """
    # Get the shape of the feature maps
    h, w, d = feature_maps.shape[1:4]
    
    # Compute the pooling size to get the desired target dimension
    pool_size = (h // int(np.sqrt(d / target_dim)), w // int(np.sqrt(d / target_dim)))
    
    # Apply average pooling
    x = tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=pool_size)(feature_maps)
    
    # Flatten the downsampled feature maps
    flattened = tf.keras.layers.Flatten()(x)
    latent = tf.keras.backend.eval(flattened)
    
    return latent

def save_latent(latent, output_path):
    """
    Save the latent representation as a numpy file.
    """
    np.save(output_path, latent)

def extract_latents_from_directory(input_dir, output_dir):
    """
    Extract and save latent representations for all images in the input directory.
    """
    # Load the VGG16 model with convolutional layers
    base_model = VGG16(weights='imagenet', include_top=False)  # Exclude the classification layer
    model = Model(inputs=base_model.input, outputs=base_model.output)
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Iterate over all files in the input directory
    input_path = Path(input_dir)
    for file_path in input_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            try:
                # Load and preprocess the image
                img_array = load_and_preprocess_image(str(file_path))
                
                # Get the feature maps
                feature_maps = get_feature_maps(model, img_array)
                
                # Downsample the feature maps to 512 dimensions
                latent = downsample_feature_maps(feature_maps)
                
                # Construct output path
                relative_path = file_path.relative_to(input_path)
                output_path = Path(output_dir) / relative_path.with_suffix('.npy')
                
                # Ensure the output directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save the latent representation
                save_latent(latent, output_path)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

if __name__ == "__main__":
    input_directory = r"C:\Users\dell\Desktop\preprocessed"  # Update with the actual path
    output_directory = r"C:\Users\dell\Desktop\latents"  # Update with the actual path
    extract_latents_from_directory(input_directory, output_directory)
