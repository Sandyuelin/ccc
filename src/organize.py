import numpy as np
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
import shutil


def load_latent_vectors(latents_dir):
    latents = {}
    for root, dirs, files in os.walk(latents_dir):
        for file in files:
            if file.endswith('.npy'):
                char_name = os.path.relpath(root, latents_dir)
                char_name = os.path.join(char_name, file)
                latent = np.load(os.path.join(root, file))
                latents[char_name] = latent
    print(f"Loaded {len(latents)} latent vectors from {latents_dir}")
    return latents


def load_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded JSON data from {json_file_path}")
    return data

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm != 0 else vector



def calculate_similarity(latent1, latent2):
    latent1_flat = normalize_vector(latent1.flatten())
    latent2_flat = normalize_vector(latent2.flatten())
    return cosine_similarity([latent1_flat], [latent2_flat])[0][0]

def find_latent_for_character(character, latents):
    for name, latent in latents.items():
        if character in name:
            return latent
    return None


def get_character_from_path(path):
    parts = path.split(os.sep)
    return parts[-2]

def organize_images_by_similarity(input_dir, output_dir, latents1, latents2, mapping):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for file_name in files:
            if file_name.endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(root, file_name)
                image_name = os.path.splitext(file_name)[0]

                image_relative_path = os.path.relpath(image_path, input_dir)
                latent_path_in_latents1 = os.path.join(original_latents_dir,
                                                       image_relative_path.replace('.jpg', '.npy').replace('.png',
                                                                                                           '.npy').replace(
                                                           '.jpeg', '.npy'))

                if os.path.exists(latent_path_in_latents1):
                    image_latent = np.load(latent_path_in_latents1)

                    character = get_character_from_path(root)
                    print(f"Processing image {file_name} with character {character}")

                    simplified_latent = find_latent_for_character(character, latents2)
                    if simplified_latent is not None:
                        simplified_similarity = calculate_similarity(image_latent, simplified_latent)
                        print(f"Simplified latent vector: {simplified_latent.flatten()[:10]}")  # Print first 10 elements for brevity
                    else:
                        simplified_similarity = -1

                    complex_similarities = []
                    if character in mapping:
                        complex_characters = mapping[character]
                        for complex_character in complex_characters:
                            complex_latent = find_latent_for_character(complex_character, latents2)
                            if complex_latent is not None:
                                complex_similarity = calculate_similarity(image_latent, complex_latent)
                                complex_similarities.append((complex_character, complex_similarity))
                                print(f"Complex character: {complex_character}, Similarity: {complex_similarity}")
                            else:
                                print(f"No latent vector found for complex character: {complex_character}")

                    if complex_similarities:
                        # Find the complex character with the highest similarity
                        best_complex_character, best_complex_similarity = max(complex_similarities, key=lambda x: x[1])
                    else:
                        best_complex_character, best_complex_similarity = None, -1

                    print(f"Simplified similarity: {simplified_similarity}, Best complex similarity: {best_complex_similarity}")

                    if simplified_similarity >= best_complex_similarity:
                        dest_folder = os.path.join(output_dir, 'simplified', character)
                    else:
                        dest_folder = os.path.join(output_dir, 'complex', best_complex_character if best_complex_character else character)

                    if not os.path.exists(dest_folder):
                        os.makedirs(dest_folder)

                    shutil.copy(image_path, os.path.join(dest_folder, file_name))
                    print(f"Copied {file_name} to {dest_folder}")
                else:
                    print(f"Latent file not found for image {file_name}. Expected at {latent_path_in_latents1}")


# Paths to your directories and files
original_latents_dir = r"C:\Users\dell\Desktop\latents"
generated_latents_dir = r"C:\Users\dell\Desktop\latents2"
json_file_path = r"C:\Users\dell\Desktop\chinese_calligraphy_classifier\simplified_complex.json"
images_dir = r"C:\Users\dell\Desktop\preprocessed"
organized_images_dir = r"C:\Users\dell\Desktop\organized_images"

# Load data
original_latents = load_latent_vectors(original_latents_dir)
generated_latents = load_latent_vectors(generated_latents_dir)
simplified_complex_mapping = load_json(json_file_path)

# Organize images by similarity
organize_images_by_similarity(images_dir, organized_images_dir, original_latents, generated_latents,
                              simplified_complex_mapping)
