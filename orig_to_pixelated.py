import cv2
import os

# Ensure the output directory exists
os.makedirs('pixelated_images', exist_ok=True)

input_dir = 'E:/unnati intel Project/original_images'
output_dir = 'E:/unnati intel Project/pixelated_images'
def create_pixelated_images(input_dir, output_dir, quality_factors=[10, 20]):
    # Check if the input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: The directory '{input_dir}' does not exist.")
        return

    # Process each image in the input directory
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)

        # Check if the image was loaded successfully
        if img is None:
            print(f"Warning: Unable to read '{img_path}'. Skipping.")
            continue

        # Create pixelated images using different quality factors
        for q in quality_factors:
            pixelated_img_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_q{q}.jpg")
            cv2.imwrite(pixelated_img_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), q])

        # Downscale and upscale to create pixelated images
        for scale in [5, 6]:
            downscaled_img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale), interpolation=cv2.INTER_NEAREST)
            upscaled_img = cv2.resize(downscaled_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            pixelated_img_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_scale{scale}.jpg")
            cv2.imwrite(pixelated_img_path, upscaled_img)

# Replace with the actual paths to your directories
create_pixelated_images('original_images', 'pixelated_images')
