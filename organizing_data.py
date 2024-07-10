import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
original_images_dir = r'E:\unnati intel Project\original_images'
pixelated_images_dir = r'E:\unnati intel Project\pixelated_images'
train_dir = r'E:\unnati intel Project\data\train'
val_dir = r'E:\unnati intel Project\data\validation'

# Create directories if they do not exist
os.makedirs(os.path.join(train_dir, 'pixelated'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'non_pixelated'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'pixelated'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'non_pixelated'), exist_ok=True)

# Function to split data and copy to train and validation directories
def split_and_copy_files(src_dir, train_dst, val_dst, test_size=0.2):
    filenames = os.listdir(src_dir)
    train_filenames, val_filenames = train_test_split(filenames, test_size=test_size, random_state=42)
    
    for filename in train_filenames:
        shutil.copy(os.path.join(src_dir, filename), train_dst)
        
    for filename in val_filenames:
        shutil.copy(os.path.join(src_dir, filename), val_dst)

# Split and copy pixelated images
split_and_copy_files(pixelated_images_dir, os.path.join(train_dir, 'pixelated'), os.path.join(val_dir, 'pixelated'))

# Split and copy non-pixelated (original) images
split_and_copy_files(original_images_dir, os.path.join(train_dir, 'non_pixelated'), os.path.join(val_dir, 'non_pixelated'))
