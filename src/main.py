from src.preprocess import preprocess_image
from src.encode import encode_image
from src.classify import calculate_cosine_similarity
import cv2

def main():
    # Paths
    raw_image_path = 'data/raw/example_image.jpg'
    font_path = 'data/fonts/standard_font.ttf'

    # Load raw image
    raw_image = cv2.imread(raw_image_path)

    # Preprocess image (in-memory)
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

if __name__ == '__main__':
    main()
