from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from utils import util
from datasets import MixedNIR2_Dai
from models import ArtifactRemovalNet
from tqdm import tqdm
from PIL import Image


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

# Initialize model and optimizer
model = ArtifactRemovalNet.RemovalNet()
model = model.to(device)  # 移动模型到 GPU
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize TensorBoard writer
model_name = model.__class__.__name__
# settings = 
# tensorboard_dir = os.path.join(work_space, f"tensorboard/{model_name+settings}")
tensorboard_dir = os.path.join(work_space, f"tensorboard/{model_name}")
os.makedirs(tensorboard_dir, exist_ok=True)
writer = SummaryWriter(tensorboard_dir)

# Initialize Checkpoint Folder
checkpoint_dir = os.path.join(work_space, f"checkpoints/{model_name}")
os.makedirs(checkpoint_dir, exist_ok=True)

# Initialize Validation output save Folder
img_save_dir = os.path.join(work_space, f"ValImages/{model_name}")
os.makedirs(img_save_dir, exist_ok=True)

# Initialize Dataset and DataLoader
batch_size = 2
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

epoch_num = 100

# Try to load the latest checkpoint
start_epoch = load_latest_checkpoint(model, optimizer, checkpoint_dir)

# Training loop
for epoch in range(start_epoch, epoch_num):  # 100 epochs as an example
    model.train()
    total_psnr = 0.0
    total_loss = 0.0
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
    
    for i, (input_images, target_images) in pbar:
        input_images = input_images.to(device)
        target_images = target_images.to(device)
        
        optimizer.zero_grad()
        
        output_images = model(input_images)
        
        loss = l1_criterion(output_images, target_images)
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
        for idx, (val_input_images, val_target_images) in enumerate(val_loader):
            val_input_images = val_input_images.to(device)
            val_target_images = val_target_images.to(device)
            
            val_output_images = model(val_input_images)
            
            val_loss_batch = l1_criterion(val_output_images, val_target_images)
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
