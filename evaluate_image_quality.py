import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import lpips
import os
import torch

def preprocess_image(img_path, size=(128, 128)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    return img

def evaluate_image_quality(original_img, restored_img):
    original_img = original_img * 2 - 1  # Normalizing between -1 and 1 for LPIPS
    restored_img = restored_img * 2 - 1  # Normalizing between -1 and 1 for LPIPS

    psnr_value = psnr(original_img, restored_img)
    ssim_value = ssim(original_img, restored_img, multichannel=True)
    lpips_value = lpips.LPIPS(net='alex').forward(
        torch.tensor(original_img).permute(2, 0, 1).unsqueeze(0),
        torch.tensor(restored_img).permute(2, 0, 1).unsqueeze(0)
    ).item()

    return psnr_value, ssim_value, lpips_value

def evaluate_dataset(original_dir, restored_dir):
    psnr_values = []
    ssim_values = []
    lpips_values = []

    for img_name in os.listdir(original_dir):
        original_img_path = os.path.join(original_dir, img_name)
        restored_img_path = os.path.join(restored_dir, img_name)

        if not os.path.exists(restored_img_path):
            continue

        original_img = preprocess_image(original_img_path)
        restored_img = preprocess_image(restored_img_path)

        psnr_value, ssim_value, lpips_value = evaluate_image_quality(original_img, restored_img)

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        lpips_values.append(lpips_value)

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_lpips = np.mean(lpips_values)

    return avg_psnr, avg_ssim, avg_lpips

# Direct paths to original and restored image directories
original_dir = 'E:\\unnati intel Project\\original_images'
restored_dir = 'E:\\unnati intel Project\\restored_images'

avg_psnr, avg_ssim, avg_lpips = evaluate_dataset(original_dir, restored_dir)

print(f"Average PSNR: {avg_psnr}, Average SSIM: {avg_ssim}, Average LPIPS: {avg_lpips}")
