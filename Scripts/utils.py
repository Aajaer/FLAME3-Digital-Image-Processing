import cv2
import numpy as np

IMG_SIZE = (224, 224)

def preprocess_image(img):
    """Resize and normalize image for the model."""
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    return img

