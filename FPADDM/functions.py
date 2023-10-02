
import torch
from skimage import morphology, filters
from inspect import isfunction
import numpy as np
from PIL import Image
from pathlib import Path
import os
import cv2


try:
    from apex import amp

    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False




def dilate_mask(mask, mode):
    if mode == "harmonization":
        element = morphology.disk(radius=7)
    if mode == "editing":
        element = morphology.disk(radius=20)
    mask = mask.permute((1, 2, 0))
    mask = mask[:, :, 0]
    mask = morphology.binary_dilation(mask, selem=element)
    mask = filters.gaussian(mask, sigma=5)
    mask = mask[:, :, None, None]
    mask = mask.transpose(3, 2, 0, 1)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask


# for roi_sampling

def stat_from_bbs(image, bb):
    y_bb, x_bb, h_bb, w_bb = bb
    bb_mean = torch.mean(image[:, :,y_bb:y_bb+h_bb, x_bb:x_bb+w_bb], dim=(2,3), keepdim=True)
    bb_std = torch.std(image[:, :, y_bb:y_bb+h_bb, x_bb:x_bb+w_bb], dim=(2,3), keepdim=True)
    return [bb_mean, bb_std]


def extract_patch(image, bb):
    y_bb, x_bb, h_bb, w_bb = bb
    image_patch = image[:, :,y_bb:y_bb+h_bb, x_bb:x_bb+w_bb]
    return image_patch


# for clip sampling
def thresholded_grad(grad, quantile=0.8):
    """
    Receives the calculated CLIP gradients and outputs the soft-tresholded gradients based on the given quantization.
    Also outputs the mask that corresponds to remaining gradients positions.
    """
    grad_energy = torch.norm(grad, dim=1)
    grad_energy_reshape = torch.reshape(grad_energy, (grad_energy.shape[0],-1))
    enery_quant = torch.quantile(grad_energy_reshape, q=quantile, dim=1, interpolation='nearest')[:,None,None] #[batch ,1 ,1]
    gead_energy_minus_energy_quant = grad_energy - enery_quant
    grad_mask = (gead_energy_minus_energy_quant > 0)[:,None,:,:]

    gead_energy_minus_energy_quant_clamp = torch.clamp(gead_energy_minus_energy_quant, min=0)[:,None,:,:]#[b,1,h,w]
    unit_grad_energy = grad / grad_energy[:,None,:,:] #[b,c,h,w]
    unit_grad_energy[torch.isnan(unit_grad_energy)] = 0
    sparse_grad = gead_energy_minus_energy_quant_clamp * unit_grad_energy #[b,c,h,w]
    return sparse_grad, grad_mask

# helper functions


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)

def read_raw_images(file_path, shape, num_images, dtype=np.uint8, big_endian=False):
    image_bytes = np.prod(shape) * np.dtype(dtype).itemsize * num_images
    with open(file_path, 'rb') as f:
        img_data = np.frombuffer(f.read(image_bytes), dtype=dtype)
    if big_endian:
        img_data = img_data.byteswap()
    images = img_data.reshape((num_images, *shape)).copy()

    return images

def generate_FPA_mask(img, thresh=50):
    outliers = np.where(img >= thresh) # if exporesure time is higher than 200ms should select larger thresh.
    mask = np.zeros_like(img, dtype=int)
    mask[outliers] = 1.0
    
    return mask

def create_img_masks(raw_path, timesteps, size=(512, 640), num_frames=268, create=False, dtype=np.uint8):

    images = read_raw_images(raw_path, size, num_frames, dtype=dtype)[0:timesteps]
    masks = []
    for idx, image in enumerate(images, 1):
        mask = generate_FPA_mask(image)
        masks.append(mask)
        if create:
            filename = "%.6d_mask.png" % (2 * idx)
            save_dir = "../datasets/FPA_masks"
            save_path = os.path.join(save_dir + filename)
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            mask_to_save = (mask.astype(np.uint8)) * 255
            cv2.imwrite(save_path, mask_to_save)

    return masks

def generate_random_even(timesteps):
    # 创建一个包含所有偶数的数组，范围是从2到2*timesteps
    even_numbers = np.arange(2, 2*timesteps + 1, 2)
    
    # 使用 numpy.random.choice 从偶数数组中随机选择一个元素
    random_even = np.random.choice(even_numbers)
    
    return random_even