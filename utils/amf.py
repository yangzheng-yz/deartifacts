from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datasets import MixedNIR2_Dai
from tqdm import tqdm
import os
from scipy.ndimage import median_filter

def read_png_image(file_path):
    with Image.open(file_path) as img:
        grayscale_img = img.convert("L")
    return np.array(grayscale_img)

def adaptive_median_filter(image, radius=2, threshold=0.2):
    """
    Apply adaptive median filter on the image.
    
    Parameters:
    - image: numpy array representing the image
    - radius: integer, the radius of the square window for the median filter
    - threshold: float, the multiplicative factor for the median to decide whether to replace a pixel
    
    Returns:
    - corrected_image: numpy array, image after applying the adaptive median filter
    """
    
    # Calculate the median image using a regular median filter
    median_image = median_filter(image, size=(2 * radius + 1))
    
    # Create a mask where pixel intensity is greater than threshold times the median in the window
    mask = (image < threshold * median_image) #  | (image < (1 - threshold) * median_image)
    
    # Create an output image initialized with original image values
    corrected_image = np.copy(image)
    
    # Replace the pixel values wherever the mask is True
    corrected_image[mask] = median_image[mask] #  * (1.1)
    
    return corrected_image

def psnr(img1, img2):
    assert img1.shape == img2.shape, "Input images must have the same dimensions."
    
    # 检查数据类型和最大值，以确定是否需要归一化
    if img1.dtype != np.float32 or np.max(img1) > 1.0:
        img1 = img1.astype(np.float32) / 255.0
    
    if img2.dtype != np.float32 or np.max(img2) > 1.0:
        img2 = img2.astype(np.float32) / 255.0
    
    mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')
    
    return 10 * np.log10(1.0 / mse)

if __name__ == "__main__":
    data_root = "/mnt/samsung/zheng_data/datasets/NIRI_to_NIRII/test_100ms/with_art"
    data_root_ = "/mnt/samsung/zheng_data/datasets/NIRI_to_NIRII/test_100ms/without_art"
    files = os.listdir(data_root)
    files.sort()
    files_path = [os.path.join(data_root, i) for i in files]
    files = os.listdir(data_root_)
    files.sort()
    files_path_ = [os.path.join(data_root_, i) for i in files]

    total_psnr = 0
    save_im_path = "/mnt/samsung/zheng_data/datasets/NIRI_to_NIRII/training_log/amf_results/test_100ms"
    os.makedirs(save_im_path, exist_ok=True)
    for idx, f_path in enumerate(files_path):
        im = read_png_image(f_path)
        im_gt = read_png_image(files_path_[idx])
        im_name = f_path.split('/')[-1]
        
        im_filtered = adaptive_median_filter(im, radius=2, threshold=0.5)
        total_psnr += psnr(im_gt, im_filtered)
        Image.fromarray(im_filtered).save(os.path.join(save_im_path, im_name))
        print("%s's PSNR is %s" % (im_name, psnr(im_gt, im_filtered)))
    print("average PSNR for total %s images is %s" % (len(files_path), total_psnr/len(files_path)))
        
