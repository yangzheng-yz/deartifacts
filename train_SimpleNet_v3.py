from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from utils import util
from datasets import MixedNIR2_Dai
from models import ArtifactRemovalNet
from tqdm import tqdm
from PIL import Image
import random
from torchvision.transforms import functional as TF

def rotate_and_scale_pair(input_image, target_image, degrees=(0, 30), scale=(1.0, 1.0)):
    angle = random.uniform(degrees[0], degrees[1])
    
    scale_factor = random.uniform(scale[0], scale[1])
    new_size = (int(input_image.shape[-2] * scale_factor), int(input_image.shape[-1] * scale_factor))

    # 旋转
    rotated_input = TF.rotate(input_image, angle)
    rotated_target = TF.rotate(target_image, angle)
    
    # 缩放
    scaled_input = TF.resize(rotated_input, new_size)
    scaled_target = TF.resize(rotated_target, new_size)
    
    return scaled_input, scaled_target

# Function to load the latest checkpoint
def load_latest_checkpoint(model, optimizer, checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        return 0  # No checkpoints available
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
    start_epoch = int(latest_checkpoint.split('_')[1].split('.')[0])
    
    checkpoint = torch.load(os.path.join(checkpoint_dir, latest_checkpoint))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return start_epoch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

work_space = "/mnt/samsung/zheng_data/datasets/NIRI_to_NIRII/training_log"
data_root = "/mnt/samsung/zheng_data/datasets/NIRI_to_NIRII"
shape = (512, 640)
artifacts_database_root = "/mnt/samsung/zheng_data/datasets/NIRI_to_NIRII/For_argumentation"
artifacts_database_path = [os.path.join(artifacts_database_root, i) for i in os.listdir(artifacts_database_root) if '100ms' not in i]
artifacts_database_path.sort()
artifacts_database = []
for path_to_raw in artifacts_database_path:
    num_images = int(path_to_raw.split('/')[-1].split('.')[0].split('_')[-2])
    raw_images = MixedNIR2_Dai.read_raw_images(path_to_raw, shape, num_images)
    artifacts_database.append(raw_images)
    
# Training settings
batch_size = 2 # 8
epoch_num = 100
use_rotate_and_scale = True
argumented_artifacts = True

# Initialize model and optimizer
model = ArtifactRemovalNet.SimpleNet()
model = model.to(device)  # 移动模型到 GPU
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize TensorBoard writer
model_name = model.__class__.__name__
# settings = 
# tensorboard_dir = os.path.join(work_space, f"tensorboard/{model_name+settings}")
tensorboard_dir = os.path.join(work_space, f"tensorboard/{model_name}_argumented_v3")
os.makedirs(tensorboard_dir, exist_ok=True)
writer = SummaryWriter(tensorboard_dir)

# Initialize Checkpoint Folder
checkpoint_dir = os.path.join(work_space, f"checkpoints/{model_name}_argumented_v3")
os.makedirs(checkpoint_dir, exist_ok=True)

# Initialize Validation output save Folder
img_save_dir = os.path.join(work_space, f"ValImages/{model_name}_argumented_v3")
os.makedirs(img_save_dir, exist_ok=True)

# Initialize Dataset and DataLoader
dataset_train = MixedNIR2_Dai.ArtifactRemovalDataset(root_dir=data_root, split='train')
dataset_val = MixedNIR2_Dai.ArtifactRemovalDataset(root_dir=data_root, split='val')
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

# Loss functions
l1_criterion = nn.L1Loss()
l2_criterion = nn.MSELoss()

# PSNR function
def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 10 * torch.log10(1 / mse)

# Try to load the latest checkpoint
start_epoch = load_latest_checkpoint(model, optimizer, checkpoint_dir)
count_argument_ms = [0,0]
# Training loop
for epoch in range(start_epoch, epoch_num):  # 100 epochs as an example
    model.train()
    total_psnr = 0.0
    total_loss = 0.0
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
    
    for i, (input_images, target_images, _) in pbar:

        if argumented_artifacts:
            if random.uniform(0,1.0) > 0.8:
                new_input_images = []
                for tgt in target_images:
                    group_num = random.randint(0,1) # 0,1,2
                    count_argument_ms[group_num] += 1
                    random_raw_group = artifacts_database[group_num]
                    random_raw = random.choice(random_raw_group)
                    threshold = 1000
                    radius = 2
                    coef = 1.0 # random.uniform(0.1,0.4)
                    new_inp = MixedNIR2_Dai.overlay_outliers_on_png(random_raw, tgt.squeeze(0), radius, threshold, coef=coef)
                    new_input_images.append(new_inp.unsqueeze(0))
                input_images = torch.stack(new_input_images)
        
        if use_rotate_and_scale:
            new_input_images, new_target_images = [], []
            for inp, tgt in zip(input_images, target_images):
                new_inp, new_tgt = rotate_and_scale_pair(inp, tgt)
                new_input_images.append(new_inp)
                new_target_images.append(new_tgt)
            input_images = torch.stack(new_input_images)
            target_images = torch.stack(new_target_images)

        input_images = input_images.to(device)
        target_images = target_images.to(device)
        
        optimizer.zero_grad()
        
        output_images = model(input_images)
        
        loss = l2_criterion(output_images, target_images)
        loss.backward()
        
        optimizer.step()
        
        batch_psnr = 0.0
        for j in range(input_images.size(0)):
            batch_psnr += psnr(output_images[j], target_images[j]).item()
        batch_psnr /= input_images.size(0)
        
        total_psnr += batch_psnr
        total_loss += loss.item()
        pbar.set_postfix({"Training Loss": loss.item(), 
                            "Training PSNR": batch_psnr})
    
    writer.add_scalar('Training Loss', total_loss / len(train_loader), epoch)
    writer.add_scalar('Training PSNR', total_psnr / len(train_loader), epoch)
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0
    pbar_val = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")

    with torch.no_grad():
        for idx, (val_input_images, val_target_images, _) in enumerate(val_loader):
            val_input_images = val_input_images.to(device)
            val_target_images = val_target_images.to(device)
            
            val_output_images = model(val_input_images)
            
            val_loss_batch = l2_criterion(val_output_images, val_target_images)
            val_loss += val_loss_batch.item()
            val_psnr_inbatch = 0.0
            for j in range(val_input_images.size(0)):
                val_psnr_inbatch += psnr(val_output_images[j], val_target_images[j]).item()
            val_psnr += val_psnr_inbatch
            pbar_val.set_postfix({"Validation Loss": val_loss_batch.item(), 
                                "Validation PSNR": val_psnr_inbatch / batch_size})
            # 转换为 numpy 数组并保存
            val_output_images_cpu = val_output_images.cpu().numpy()
            val_ori_images_cpu = val_input_images.cpu().numpy()
            val_target_images_cpu = val_target_images.cpu().numpy()
            for j in range(val_output_images_cpu.shape[0]):
                img_array_output = val_output_images_cpu[j, 0]  # 获取第 j 张图像，假设单通道
                img_array_output = (img_array_output * 255).astype('uint8')  # 如果需要，进行缩放和类型转换
                img_output = Image.fromarray(img_array_output, 'L')  # 创建 PIL 图像
                img_output.save(os.path.join(img_save_dir, f"val_epoch_{idx * batch_size + j}_outputimg.png"))

                img_array_ori = val_ori_images_cpu[j, 0]  # 获取第 j 张图像，假设单通道
                img_array_ori = (img_array_ori * 255).astype('uint8')  # 如果需要，进行缩放和类型转换
                img_ori = Image.fromarray(img_array_ori, 'L')  # 创建 PIL 图像
                img_ori.save(os.path.join(img_save_dir, f"val_epoch_{idx * batch_size + j}_oriimg.png"))

                img_array_target = val_target_images_cpu[j, 0]  # 获取第 j 张图像，假设单通道
                img_array_target = (img_array_target * 255).astype('uint8')  # 如果需要，进行缩放和类型转换
                img_target = Image.fromarray(img_array_target, 'L')  # 创建 PIL 图像
                img_target.save(os.path.join(img_save_dir, f"val_epoch_{idx * batch_size + j}_targetimg.png"))

    avg_val_loss = val_loss / len(val_loader)
    avg_val_psnr = val_psnr / len(val_loader) / val_input_images.size(0)

    # # Update progress bar
    # pbar.set_postfix({"Training Loss": total_loss / (i + 1), 
    #                     "Training PSNR": total_psnr / (i + 1), 
    #                     "Validation Loss": avg_val_loss, 
    #                     "Validation PSNR": avg_val_psnr})
    
    writer.add_scalar('Validation Loss', avg_val_loss, epoch)
    writer.add_scalar('Validation PSNR', avg_val_psnr, epoch)
    
    # Save model weights
    model_save_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_save_path)
print("total %s times 20ms argumentation" % count_argument_ms[0])
print("total %s times 200ms argumentation" % count_argument_ms[1])
