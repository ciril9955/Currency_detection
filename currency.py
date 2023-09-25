
import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import ResNet50
from sklearn.metrics.pairwise import cosine_similarity

# Constants
UPLOADS_DIR = "currencyapi/test_img"
ORIGINAL_DATASET_DIR = "currencyapi/orginal/orginal"
FAKE_DATASET_DIR = "currencyapi/fake/fake"
MAX_UPLOADS = 5
ORIGINAL_SIMILARITY_THRESHOLD = 0.80
FAKE_SIMILARITY_THRESHOLD = 0.90

# Function to preprocess an image
def preprocess_image(image_path):
    try:
        # Open and convert the image to RGB mode
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224))  # Resize the image to the desired size
        img = img_to_array(img)
        img = img / 255.0
        return img
    except Exception as e:
        print(f"Error: Image at {image_path} cannot be read or is not in a supported format.")
        print(f"Error message: {str(e)}")
        return None

# Function to load and preprocess images from a directory
def load_and_preprocess_images(directory):
    images = []
    for img_file in os.listdir(directory):
        img_path = os.path.join(directory, img_file)
        img = preprocess_image(img_path)
        if img is not None:
            images.append(img)
    return np.array(images)

# Decorator to choose the latest uploaded image for testing
def with_latest_uploaded_image(func):
    def wrapper(*args, **kwargs):
        uploaded_image_paths = [os.path.join(UPLOADS_DIR, img_file) for img_file in os.listdir(UPLOADS_DIR)]
        if uploaded_image_paths:
            latest_uploaded_image = max(uploaded_image_paths, key=os.path.getctime)
            kwargs['latest_uploaded_image'] = latest_uploaded_image
        else:
            kwargs['latest_uploaded_image'] = None
        return func(*args, **kwargs)
    return wrapper

# Decorator to handle deletion of the oldest uploaded image if needed
def with_image_cleanup(func):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        uploaded_image_paths = [os.path.join(UPLOADS_DIR, img_file) for img_file in os.listdir(UPLOADS_DIR)]
        if len(uploaded_image_paths) > MAX_UPLOADS:
            oldest_image_path = min(uploaded_image_paths, key=os.path.getctime)
            os.remove(oldest_image_path)
    return wrapper

# Function to calculate cosine similarity between two sets of images
def calculate_similarity(images_set1, images_set2):
    similarities = []
    for img1 in images_set1:
        for img2 in images_set2:
            similarity = cosine_similarity(img1.reshape(1, -1), img2.reshape(1, -1))
            similarities.append(similarity[0][0])
    return similarities

# Function to determine if an uploaded image is original or fake currency
@with_latest_uploaded_image
@with_image_cleanup
def classify_uploaded_image(latest_uploaded_image):
    if latest_uploaded_image is None:
        print("Error: No uploaded images found in the directory.")
        return

    original_images = load_and_preprocess_images(ORIGINAL_DATASET_DIR)
    fake_images = load_and_preprocess_images(FAKE_DATASET_DIR)

    similarity_original = calculate_similarity(original_images, [preprocess_image(latest_uploaded_image)])
    similarity_fake = calculate_similarity(fake_images, [preprocess_image(latest_uploaded_image)])
    
    avg_similarity_original = np.mean(similarity_original)
    avg_similarity_fake = np.mean(similarity_fake)

    print("Average Cosine Similarity with original dataset:", avg_similarity_original)
    print("Average Cosine Similarity with fake dataset:", avg_similarity_fake)

    if avg_similarity_original > ORIGINAL_SIMILARITY_THRESHOLD:
        print("The uploaded image is an original currency.")
    elif avg_similarity_fake < FAKE_SIMILARITY_THRESHOLD:
        print("The uploaded image is a fake currency.")
    else:
        print("The authenticity of the uploaded image cannot be determined.")

# Call the classify_uploaded_image function directly
classify_uploaded_image()
