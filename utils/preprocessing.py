import cv2
import numpy as np

def preprocess_image(img_path, img_size=224):
    """
    Reads a retinal fundus image, applies preprocessing,
    and returns a normalized image array.
    """

    # Read image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found or cannot be read: {img_path}")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img = cv2.resize(img, (img_size, img_size))

    # Apply CLAHE on L-channel (LAB color space)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Normalize to [0,1]
    img = img.astype("float32") / 255.0

    return img
