import os
import cv2
import numpy as np
import tensorflow as tf

def preprocess_image(img_path, size=(128, 128)):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image: {img_path}")
        return None
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    return img

def save_image(img, path):
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(path, img)

def generate_restored_images(pixelated_dir, restored_dir, model_path):
    if not os.path.exists(restored_dir):
        os.makedirs(restored_dir)

    model = tf.keras.models.load_model(model_path)

    for img_name in os.listdir(pixelated_dir):
        pixelated_img_path = os.path.join(pixelated_dir, img_name)
        restored_img_path = os.path.join(restored_dir, img_name)

        pixelated_img = preprocess_image(pixelated_img_path)
        if pixelated_img is None:
            continue

        restored_img = model.predict(np.expand_dims(pixelated_img, axis=0))[0]
        save_image(restored_img, restored_img_path)

# Direct paths to pixelated and restored image directories
pixelated_dir = 'E:\\unnati intel Project\\pixelated_images'
restored_dir = 'E:\\unnati intel Project\\restored_images'
model_path = 'E:\\unnati intel Project\\best_improved_cnn_model_v2.keras'

generate_restored_images(pixelated_dir, restored_dir, model_path)
