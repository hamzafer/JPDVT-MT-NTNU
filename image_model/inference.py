#!/usr/bin/env python
"""
JPDVT Inference on Test Dataset

Processes all JPEG images in a test folder, performs jigsaw puzzle reconstruction 
with variable grid sizes, calculates accuracy, and saves outputs into a structured
inference directory. Logs high-level info and errors without interrupting the run.
"""

import os
import time
import glob
import logging
import torch
import torchvision
from torchvision.utils import save_image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from einops import rearrange
from sklearn.metrics import pairwise_distances
import argparse

# Custom modules (make sure PYTHONPATH includes the directory with these)
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models import DiT_models, get_2d_sincos_pos_embed

# ================================
# Constants / Default Parameters
# ================================
DEFAULT_IMAGE_SIZE = 192
DEFAULT_GRID_SIZE = 3
DEFAULT_NUM_SAMPLING_STEPS = 250
DEFAULT_SEED = 0
DEFAULT_MODEL = "JPDVT"
DEFAULT_DATASET = "imagenet"
BASE_DATA_PATH = "/cluster/home/muhamhz/data/imagenet/test"  # test images directory
DEFAULT_CKPT = "/cluster/home/muhamhz/JPDVT/image_model/results/009-imagenet-JPDVT-crop/checkpoints/2850000.pt"

# Output directories
INFERENCE_DIR = "/cluster/home/muhamhz/JPDVT/image_model/inference"
LOGS_DIR = "/cluster/home/muhamhz/JPDVT/image_model/logs"

# Create directories if they don't exist
os.makedirs(INFERENCE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ================================
# Logging Setup
# ================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGS_DIR, "inference.log"), mode='w')
    ]
)
ERROR_LOG = os.path.join(LOGS_DIR, "error_images.txt")

# ================================
# Utility Functions
# ================================

def setup_transform(image_size):
    """Create a transform that center-crops to image_size, converts to tensor and normalizes."""
    def center_crop_arr(pil_image, image_size):
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])
    
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3, inplace=True)
    ])
    return transform

def load_image(image_path, transform):
    """Load an image and apply transform; raises exception if fails."""
    try:
        img = Image.open(image_path).convert("RGB")
        return transform(img)
    except Exception as e:
        logging.error(f"Failed to load image {image_path}: {e}")
        raise

# ================================
# Processing a Single Image
# ================================
def process_image(image_tensor, filename, args, model, diffusion, time_emb, time_emb_noise, device, grid_size):
    """
    Process one image:
      - Save original image.
      - Split into grid patches and scramble using a random permutation.
      - Run diffusion inference and predict permutation.
      - Reconstruct the final puzzle.
      - Save scrambled and reconstructed images.
    Returns: (correct_flag, predicted permutation, ground truth permutation)
    """
    # Add batch dimension: shape becomes [1, C, H, W]
    x = image_tensor.unsqueeze(0).to(device)
    
    # Create output subdir for this grid size
    grid_dir = os.path.join(INFERENCE_DIR, str(grid_size))
    os.makedirs(grid_dir, exist_ok=True)
    
    # Save original image
    original_save_path = os.path.join(grid_dir, f"{os.path.splitext(filename)[0]}_original.png")
    save_image(x, original_save_path, normalize=True)
    
    # Split image into grid patches
    patch_size = args.image_size // grid_size
    # Rearranging: from [B, C, H, W] to [B, C, grid_size^2, patch_size, patch_size]
    x_grid = rearrange(x, 'b c (p1 h1) (p2 w1) -> b c (p1 p2) h1 w1', 
                         p1=grid_size, p2=grid_size, h1=patch_size, w1=patch_size)
    
    num_patches = grid_size * grid_size
    # Generate random permutation indices and scramble patches
    true_indices = np.random.permutation(num_patches)
    scrambled = x_grid.clone()
    scrambled = scrambled[:, :, true_indices, :, :]
    
    # Reassemble scrambled patches into full image
    scrambled_reassembled = rearrange(scrambled, 'b c (p1 p2) h1 w1 -> b c (p1 h1) (p2 w1)', 
                                      p1=grid_size, p2=grid_size, h1=patch_size, w1=patch_size)
    scrambled_save_path = os.path.join(grid_dir, f"{os.path.splitext(filename)[0]}_scrambled.png")
    save_image(scrambled_reassembled, scrambled_save_path, normalize=True)
    
    # Run diffusion inference on scrambled image
    model.train()  # necessary for batchnorm if batch size is 1
    samples = diffusion.p_sample_loop(
        model.forward, scrambled_reassembled, time_emb_noise.shape, time_emb_noise,
        clip_denoised=False, model_kwargs=None, progress=False, device=device
    )
    
    # Process the diffusion sample to predict permutation.
    # Determine the sample patch dimension (this factor may be adjusted as needed)
    sample_patch_dim = args.image_size // (16 * grid_size)
    sample = samples[0]  # assume samples has one element for batch size 1
    sample_rearr = rearrange(sample, f'(p1 h1 p2 w1) d -> (p1 p2) (h1 w1) d', 
                               p1=grid_size, p2=grid_size, h1=sample_patch_dim, w1=sample_patch_dim)
    sample_rearr = sample_rearr.mean(1)  # shape: [num_patches, d]
    dist = pairwise_distances(sample_rearr.cpu().numpy(), time_emb[0].cpu().numpy(), metric='manhattan')
    
    # Greedy algorithm to predict permutation
    def find_permutation(distance_matrix):
        sort_list = []
        for m in range(distance_matrix.shape[1]):
            order = distance_matrix[:, 0].argmin()
            sort_list.append(order)
            distance_matrix = distance_matrix[:, 1:]
            distance_matrix[order, :] = 2024
        return sort_list
    
    order = find_permutation(dist.copy())
    pred = np.asarray(order).argsort()
    
    # Compare prediction to true scramble
    correct = int((pred == true_indices).all())
    
    # Reconstruct final puzzle image based on predicted permutation
    scrambled_patches = [scrambled[0, :, i, :, :] for i in range(num_patches)]
    reconstructed_patches = [None] * num_patches
    for i, pos in enumerate(pred):
        reconstructed_patches[pos] = scrambled_patches[i]
    grid_reconstructed = torch.stack(reconstructed_patches)
    final_reconstructed = rearrange(grid_reconstructed, ' (p1 p2) c h1 w1 -> c (p1 h1) (p2 w1)', 
                                    p1=grid_size, p2=grid_size, h1=patch_size, w1=patch_size)
    
    # Save final reconstructed image with accuracy in filename
    final_save_path = os.path.join(grid_dir, f"{os.path.splitext(filename)[0]}_reconstructed_acc{correct}.png")
    save_image(final_reconstructed.unsqueeze(0), final_save_path, normalize=True)
    
    return correct, pred, true_indices

# ================================
# Main Inference Loop
# ================================
def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup transform and log
    transform = setup_transform(args.image_size)
    logging.info(f"Using test data from: {args.data_path}")
    
    # Load model
    model = DiT_models[args.model](input_size=args.image_size).to(device)
    logging.info(f"Loading model from {args.ckpt}")
    model_dict = model.state_dict()
    state_dict = torch.load(args.ckpt, weights_only=False)
    model_state_dict = state_dict['model']
    pretrained_dict = {k: v for k, v in model_state_dict.items() if k in model_dict}
    model.load_state_dict(pretrained_dict, strict=False)
    logging.info("Model keys (sample): " + str(list(model_dict.keys())[:10]))
    
    model.train()  # for batchnorm
    
    # Create diffusion process
    diffusion = create_diffusion(str(args.num_sampling_steps))
    
    # Setup time embeddings (matching grid size)
    time_emb = torch.tensor(get_2d_sincos_pos_embed(8, args.grid_size)).unsqueeze(0).float().to(device)
    # Increase number of embeddings for noise if needed (here using grid_size*4)
    time_emb_noise = torch.tensor(get_2d_sincos_pos_embed(8, args.grid_size*4)).unsqueeze(0).float().to(device)
    time_emb_noise = torch.randn_like(time_emb_noise).repeat(1,1,1)
    
    # List all JPEG images in the test directory
    image_paths = glob.glob(os.path.join(args.data_path, "*.JPEG"))
    logging.info(f"Found {len(image_paths)} test images.")
    
    total = 0
    correct_total = 0
    start_time = time.time()
    
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        try:
            logging.info(f"Processing {filename} ...")
            img_tensor = load_image(image_path, transform)
            t0 = time.time()
            correct, pred, true_indices = process_image(img_tensor, filename, args, model, diffusion,
                                                          time_emb, time_emb_noise, device, args.grid_size)
            t1 = time.time()
            inference_time = t1 - t0
            logging.info(f"{filename}: Predicted order: {pred}, True order: {true_indices}, Accuracy: {correct}, Time: {inference_time:.2f} sec")
            total += 1
            correct_total += correct
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")
            with open(ERROR_LOG, "a") as f:
                f.write(f"{filename}: {e}\n")
            continue
    
    overall_accuracy = correct_total / total if total > 0 else 0
    total_time = time.time() - start_time
    logging.info(f"Processed {total} images with overall accuracy: {overall_accuracy:.4f}")
    logging.info(f"Total inference time: {total_time:.2f} sec, Average per image: {total_time/total:.2f} sec")

# ================================
# Entry Point
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=str, choices=["imagenet", "met"], default=DEFAULT_DATASET)
    parser.add_argument("--data-path", type=str, default=BASE_DATA_PATH, help="Directory containing test images")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--num-sampling-steps", type=int, default=DEFAULT_NUM_SAMPLING_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT, help="Checkpoint path")
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE, help="Grid size (e.g., 3 for 3x3, 4 for 4x4)")
    
    args = parser.parse_args()
    main(args)
