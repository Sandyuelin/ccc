import torch
import clip
from torchvision import models, transforms
from PIL import Image

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess_clip = clip.load("ViT-B/32", device=device)

# Load VGG-16 model
vgg16_model = models.vgg16(pretrained=True).to(device)
vgg16_model.eval()

# Load Inception-v3 model
inception_model = models.inception_v3(pretrained=True).to(device)
inception_model.eval()

# Preprocessing function for VGG-16
preprocess_vgg16 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Preprocessing function for Inception-v3
preprocess_inception = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def encode_with_clip(image):
    image = preprocess_clip(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
    return image_features.cpu().numpy()

def encode_with_vgg16(image):
    image = preprocess_vgg16(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = vgg16_model(image)
    return features.cpu().numpy()

def encode_with_inception(image):
    image = preprocess_inception(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = inception_model(image)
    return features.cpu().numpy()

def encode_image(image, method="clip"):
    if method == "clip":
        return encode_with_clip(image)
    elif method == "vgg16":
        return encode_with_vgg16(image)
    elif method == "inception":
        return encode_with_inception(image)
    else:
        raise ValueError("Unsupported encoding method")

# Example usage
if __name__ == "__main__":
    # Load and preprocess the image
    raw_image_path = 'data/raw/example_image.jpg'
    raw_image = Image.open(raw_image_path).convert('RGB')

    # Encode with CLIP
    clip_features = encode_image(raw_image, method="clip")
    print("CLIP features:", clip_features)

    # Encode with VGG-16
    vgg16_features = encode_image(raw_image, method="vgg16")
    print("VGG-16 features:", vgg16_features)

    # Encode with Inception-v3
    inception_features = encode_image(raw_image, method="inception")
    print("Inception-v3 features:", inception_features)