import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torchvision
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models import DiT_models
import argparse
from torch.utils.data import DataLoader
from models import get_2d_sincos_pos_embed
from datasets import MET
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from collections import defaultdict
import random

# ======= Configuration for TEXMET =======
MODEL_NAME = "JPDVT"
DATASET_NAME = "texmet"
BASE_DATA_PATH = "/cluster/home/muhamhz/data/TEXMET/images/"
DATA_PATH = ""  # All images are in BASE_DATA_PATH
CROP = False
IMAGE_SIZE = 288
NUM_SAMPLING_STEPS = 250
SEED = 0
CKPT_PATH = "/cluster/home/muhamhz/JPDVT/image_model/results/002-texmet-JPDVT-crop/checkpoints/manual_1752163781-parsit.pt"
GRID_SIZE = 3
CSV_PATH = "/cluster/home/muhamhz/JPDVT/image_model/logs/inference_progress.csv"

FULL_DATA_PATH = BASE_DATA_PATH  # No subfolder for TEXMET test

print("Configuration loaded for TEXMET.")

def imshow_tensor(img_tensor, title="", ax=None):
    """Display a PyTorch tensor as an image"""
    img_tensor = img_tensor * 0.5 + 0.5
    npimg = img_tensor.permute(1, 2, 0).cpu().numpy()
    if ax is None:
        plt.imshow(np.clip(npimg, 0, 1))
        plt.title(title)
        plt.axis("off")
        plt.show()
    else:
        ax.imshow(np.clip(npimg, 0, 1))
        ax.set_title(title)
        ax.axis("off")

def center_crop_arr(pil_image, image_size):
    """Center cropping implementation from ADM"""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def find_permutation(distance_matrix):
    """Greedy algorithm to find permutation order"""
    sort_list = []
    distance_matrix_copy = distance_matrix.copy()
    for _ in range(distance_matrix.shape[1]):
        order = distance_matrix_copy[:, 0].argmin()
        sort_list.append(order)
        distance_matrix_copy = distance_matrix_copy[:, 1:]
        distance_matrix_copy[order, :] = 2024
    return sort_list

def load_and_process_image(image_path, transform):
    """Load and process a single image"""
    try:
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None
        
        pil_image = Image.open(image_path).convert('RGB')
        tensor = transform(pil_image)
        return tensor.unsqueeze(0)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def create_scrambled_image(x, indices, G, IMAGE_SIZE):
    """Create scrambled image given original image and permutation indices"""
    x_patches = rearrange(
        x, 'b c (p1 h1) (p2 w1) -> b c (p1 p2) h1 w1', 
        p1=G, p2=G, h1=IMAGE_SIZE//G, w1=IMAGE_SIZE//G
    )
    
    x_patches = x_patches[:, :, indices, :, :]
    
    x_scrambled = rearrange(
        x_patches, 'b c (p1 p2) h1 w1 -> b c (p1 h1) (p2 w1)', 
        p1=G, p2=G, h1=IMAGE_SIZE//G, w1=IMAGE_SIZE//G
    )
    
    return x_scrambled, x_patches

def reconstruct_puzzle(scrambled_patches, predicted_order, G):
    """Reconstruct puzzle using predicted ordering"""
    reconstructed_patches = [None] * (G * G)
    for i, pos in enumerate(predicted_order):
        reconstructed_patches[pos] = scrambled_patches[i]
    
    return torch.stack(reconstructed_patches)

def run_inference_on_image(x, model, diffusion, time_emb_noise, G, IMAGE_SIZE):
    """Run inference on a single image and return results"""
    indices = np.random.permutation(G * G)
    
    x_scrambled, x_patches = create_scrambled_image(x, indices, G, IMAGE_SIZE)
    
    scrambled_patches = [x_patches[0, :, i, :, :] for i in range(G * G)]
    
    samples = diffusion.p_sample_loop(
        model.forward, 
        x_scrambled, 
        time_emb_noise.shape, 
        time_emb_noise, 
        clip_denoised=False, 
        model_kwargs=None, 
        progress=False, 
        device=device
    )
    
    for sample in samples:
        sample_patch_dim = IMAGE_SIZE // (16 * G)
        
        sample = rearrange(
            sample, '(p1 h1 p2 w1) d -> (p1 p2) (h1 w1) d', 
            p1=G, p2=G, h1=sample_patch_dim, w1=sample_patch_dim
        )
        
        sample = sample.mean(1)
        
        dist = pairwise_distances(sample.cpu().numpy(), time_emb[0].cpu().numpy(), metric='manhattan')
        order = find_permutation(dist)
        pred = np.asarray(order).argsort()
        
        puzzle_correct = int((pred == indices).all())
        patch_matches = int((pred == indices).sum())
        
        reconstructed_grid = reconstruct_puzzle(scrambled_patches, pred, G)
        
        return {
            'original': x[0],
            'scrambled': x_scrambled[0],
            'reconstructed_grid': reconstructed_grid,
            'true_indices': indices,
            'pred_indices': pred,
            'puzzle_correct': puzzle_correct,
            'patch_matches': patch_matches,
            'scrambled_patches': scrambled_patches
        }

# Load and analyze CSV results
df = pd.read_csv(CSV_PATH)

# Randomly select a perfect and a failed example each run
perfect_cases = df[df['puzzle_correct'] == 1].sample(n=1)
failed_cases = df[df['puzzle_correct'] == 0].sample(n=1)

perfect_filename = perfect_cases.iloc[0]['filename']
failed_filename = failed_cases.iloc[0]['filename']

print("Perfect example:", perfect_filename)
print("Failed example:", failed_filename)

# Setup model and device
torch.manual_seed(SEED)
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = DiT_models[MODEL_NAME](input_size=IMAGE_SIZE).to(device)
state_dict = torch.load(CKPT_PATH, weights_only=False)
model_state_dict = state_dict['model']

model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in model_state_dict.items() if k in model_dict}
model.load_state_dict(pretrained_dict, strict=False)
model.train()

diffusion = create_diffusion(str(NUM_SAMPLING_STEPS))

# Setup transforms and embeddings
transform = transforms.Compose([
    transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
])

G = GRID_SIZE
time_emb = torch.tensor(get_2d_sincos_pos_embed(8, G)).unsqueeze(0).float().to(device)
PATCHES_PER_SIDE = IMAGE_SIZE // 16  # 288//16 = 18
time_emb_noise = torch.tensor(get_2d_sincos_pos_embed(8, PATCHES_PER_SIDE)).unsqueeze(0).float().to(device)
time_emb_noise = torch.randn_like(time_emb_noise).repeat(1, 1, 1)

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Perfect case
perfect_path = os.path.join(FULL_DATA_PATH, perfect_filename)
perfect_img = load_and_process_image(perfect_path, transform)

if perfect_img is not None:
    perfect_img = perfect_img.to(device)
    perfect_result = run_inference_on_image(perfect_img, model, diffusion, time_emb_noise, G, IMAGE_SIZE)
    
    # Original
    imshow_tensor(perfect_result['original'], "Original", axes[0, 0])
    
    # Scrambled
    imshow_tensor(perfect_result['scrambled'], "Scrambled", axes[0, 1])
    
    # Reconstructed
    recon_grid = torchvision.utils.make_grid(perfect_result['reconstructed_grid'], nrow=G, normalize=True)
    axes[0, 2].imshow(recon_grid.permute(1, 2, 0).cpu().numpy())
    axes[0, 2].set_title(f"Reconstructed\nPuzzle: {perfect_result['puzzle_correct']}, Patch: {perfect_result['patch_matches']/(G*G):.2f}")
    axes[0, 2].axis("off")

# Failed case
print(failed_cases)
failed_path = os.path.join(FULL_DATA_PATH, failed_filename)
failed_img = load_and_process_image(failed_path, transform)

if failed_img is not None:
    failed_img = failed_img.to(device)
    failed_result = run_inference_on_image(failed_img, model, diffusion, time_emb_noise, G, IMAGE_SIZE)
    
    # Original
    imshow_tensor(failed_result['original'], "Original", axes[1, 0])
    
    # Scrambled
    imshow_tensor(failed_result['scrambled'], "Scrambled", axes[1, 1])
    
    # Reconstructed
    recon_grid = torchvision.utils.make_grid(failed_result['reconstructed_grid'], nrow=G, normalize=True)
    axes[1, 2].imshow(recon_grid.permute(1, 2, 0).cpu().numpy())
    axes[1, 2].set_title(f"Reconstructed\nPuzzle: {failed_result['puzzle_correct']}, Patch: {failed_result['patch_matches']/(G*G):.2f}")
    axes[1, 2].axis("off")

plt.suptitle("JPDVT 3x3 Perfect v failed example", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
