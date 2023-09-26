import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
from utils import util
from datasets import MixedNIR2_Dai
from models import ArtifactRemovalNet
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# Initialize Dataset and DataLoader for the test set
batch_size = 1  # Change this based on your available resources
data_root = "/mnt/samsung/zheng_data/datasets/NIRI_to_NIRII"
dataset_test = MixedNIR2_Dai.ArtifactRemovalDataset(root_dir=data_root, split='test_20ms')
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize model and load checkpoint
model = ArtifactRemovalNet.SimpleNet()
checkpoint_path = "/mnt/samsung/zheng_data/datasets/NIRI_to_NIRII/training_log/checkpoints/SimpleNet_argumented_v2/best.pth"  # Replace this with the path to your checkpoint file
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Output directories
output_image_dir = "/mnt/samsung/zheng_data/datasets/NIRI_to_NIRII/training_log/test_20ms_SimpleNet_argu0.2"
os.makedirs(output_image_dir, exist_ok=True)

# PSNR function
def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 10 * torch.log10(1 / mse)

# Test the model
total_psnr = 0.0

with torch.no_grad():
    for i, (input_images, target_images, image_name) in enumerate(tqdm(test_loader, desc="Testing")):
        input_images = input_images.to(device)
        target_images = target_images.to(device)
        
        output_images = model(input_images)
        
        # Compute and accumulate PSNR
        batch_psnr = 0.0
        for j in range(input_images.size(0)):
            batch_psnr += psnr(output_images[j], target_images[j]).item()
        batch_psnr /= input_images.size(0)
        total_psnr += batch_psnr
        print("%s's PSNR is %s" % (image_name, batch_psnr))
        # Save the output images
        for j in range(output_images.size(0)):
            save_image(output_images[j], os.path.join(output_image_dir, f"{image_name}_output.png"))
            save_image(input_images[j], os.path.join(output_image_dir, f"{image_name}_input.png"))
            save_image(target_images[j], os.path.join(output_image_dir, f"{image_name}_target.png"))

# Compute and print the average PSNR across the test set
average_psnr = total_psnr / len(test_loader)
print(f"The average PSNR on the test set is: {average_psnr:.2f} dB")
