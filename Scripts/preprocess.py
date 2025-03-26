import os
import cv2
import numpy as np

IMG_SIZE = (224, 224)
DATASET_PATH = DATASET_PATH = "/home/aajaer/FireDetection/Datasets/CVSubset/FLAME3_CVDataset/"

def load_images():
    """Load and preprocess images from the dataset."""
    images, labels = [], []
    for label, folder in enumerate(["Fire", "NoFire"]):  # Ensure "No Fire" has a space
        category_path = os.path.join(DATASET_PATH, folder, "RGB", "CorrectedFOV")
        if not os.path.exists(category_path):
            print(f"Path does not exist: {category_path}")
            continue
        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, IMG_SIZE) / 255.0  # Normalize
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

def preprocess_image(img):
    """Resize and normalize a single image."""
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    return img

