# Importing required libraries
import os
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter
# Importing required libraries for file manipulation
import glob
import random
from torch.utils.data import Dataset
import torch

# Function to read raw images
def read_raw_images(file_path, shape, num_images, dtype=np.int16, big_endian=False):
    image_bytes = np.prod(shape) * np.dtype(dtype).itemsize * num_images
    with open(file_path, 'rb') as f:
        img_data = np.frombuffer(f.read(image_bytes), dtype=dtype)
    if big_endian:
        img_data = img_data.byteswap()
    images = img_data.reshape((num_images, *shape))
    return images

# Function to read PNG images
def read_png_image(file_path):
    with Image.open(file_path) as img:
        grayscale_img = img.convert("L")
    return np.array(grayscale_img)

# Function to overlay outliers on PNG images
def overlay_outliers_on_png(raw_image, png_image, radius=2, threshold=50):
    median_image = median_filter(raw_image, size=(2 * radius + 1))
    diff = np.abs(raw_image - median_image)
    
    outliers = np.where(diff > threshold)
    
    overlay_image = png_image.copy()
    overlay_image[outliers] = 0
    return overlay_image

# Function to save image pairs with artifacts
def save_image_with_artifacts(with_art_image, save_dir, img_name):
    with_art_path = os.path.join(save_dir, 'with_art', img_name)
    Image.fromarray(with_art_image).save(with_art_path)

# Function to load image pairs
def load_image_pair(load_dir, img_idx):
    with_art_path = os.path.join(load_dir, 'with_art', f"{img_idx:04d}.png")
    without_art_path = os.path.join(load_dir, 'without_art', f"{img_idx:04d}.png")
    
    with_art_image = read_png_image(with_art_path)
    without_art_image = read_png_image(without_art_path)
    
    return with_art_image, without_art_image

class ArtifactRemovalDataset(Dataset):
    def __init__(self, root_dir, split='train', training=True):
        self.root_dir = root_dir
        self.split = split
        self.training = training
        
        self.with_art_dir = os.path.join(root_dir, split, 'with_art')
        self.without_art_dir = os.path.join(root_dir, split, 'without_art')
        
        # Assuming the images are named as 0000.png, 0001.png, and so on.
        self.image_files = sorted(os.listdir(self.without_art_dir))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        with_art_path = os.path.join(self.with_art_dir, img_name)
        without_art_path = os.path.join(self.without_art_dir, img_name)
        
        with_art_image = read_png_image(with_art_path)
        without_art_image = read_png_image(without_art_path)
        
        # Convert to PyTorch tensors
        with_art_tensor = torch.tensor(with_art_image[np.newaxis, :, :], dtype=torch.float32) / 255.0
        without_art_tensor = torch.tensor(without_art_image[np.newaxis, :, :], dtype=torch.float32) / 255.0
        
        return with_art_tensor, without_art_tensor

# Function to generate the dataset with and without artifacts
def generate_artifact_dataset(data_root, path_to_raw, shape, num_images, radius=2, threshold=50):
    for dataset_type in ['train', 'test', 'val']:
        # Directory where the images will be saved
        save_dir = os.path.join(data_root, dataset_type)
        
        # Make sure the with_art directories exist
        os.makedirs(os.path.join(save_dir, 'with_art'), exist_ok=True)
        
        # List all PNG images in the with_art folder
        png_files = glob.glob(os.path.join(save_dir, 'without_art', '*.png'))
        
        # Read raw images from the raw file
        raw_images = read_raw_images(path_to_raw, shape, num_images)
        
        for img_idx, png_file in enumerate(png_files):
            # Read the PNG image
            png_image = read_png_image(png_file)
            
            # Randomly choose one raw image to generate artifact
            random_raw = random.choice(raw_images)
            
            # Generate image with artifacts
            threshold = random.randint(1000, 2000)
            with_art_image = overlay_outliers_on_png(random_raw, png_image, radius, threshold)
            
            # Save the image pair
            save_image_with_artifacts(with_art_image, save_dir, png_file.split('/')[-1])

if __name__ == "__main__":
    # Example usage
    # Assuming data_root is the path where 'train', 'test', and 'val' folders are located
    # Assuming path_to_raw is the path to the .raw file containing multiple raw images
    # Assuming shape is the shape of each raw image (e.g., (256, 256))
    # Assuming num_images is the number of raw images in the .raw file

    # Uncomment and modify the following lines to run the function
    data_root = "/mnt/samsung/zheng_data/datasets/NIRI_to_NIRII"
    path_to_raw = "/mnt/samsung/zheng_data/datasets/100ms.raw"
    shape = (512, 640)
    num_images = 10

    generate_artifact_dataset(data_root, path_to_raw, shape, num_images, radius=2, threshold=1000)

