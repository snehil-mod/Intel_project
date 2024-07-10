import os
import cv2
import numpy as np

def is_pixelated(image_path, threshold=0.1):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return False

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate the Laplacian variance as a measure of pixelation
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Compare Laplacian variance with threshold
    if lap_var < threshold:
        return True  # Image is considered pixelated
    else:
        return False  # Image is not pixelated

# Directory paths to pixelated images
pixelated_dir = r'E:\unnati intel Project\pixelated_images'

# Iterate through images in the directory
for img_name in os.listdir(pixelated_dir):
    image_path = os.path.join(pixelated_dir, img_name)
    if is_pixelated(image_path):
        print(f"The image '{img_name}' at '{image_path}' is pixelated.")
    else:
        print(f"The image '{img_name}' at '{image_path}' is not pixelated.")
