import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(features1, features2):
    return cosine_similarity(features1, features2)

# Example usage
if __name__ == "__main__":
    from preprocess import preprocess_image
    from encode import encode_image
    import cv2

    # Load and preprocess the image
    raw_image_path = '../data/raw/example_image.jpg'
    raw_image = cv2.imread(raw_image_path)
    preprocessed_image = preprocess_image(raw_image)

    # Encode with CLIP
    clip_features = encode_image(preprocessed_image, method="clip")
    print("CLIP features:", clip_features)

    # Encode with VGG-16
    vgg16_features = encode_image(preprocessed_image, method="vgg16")
    print("VGG-16 features:", vgg16_features)

    # Calculate cosine similarity
    similarity = calculate_cosine_similarity(clip_features, vgg16_features)
    print("Cosine similarity:", similarity)
