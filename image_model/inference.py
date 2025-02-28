#!/usr/bin/env python3
"""
Inference script to solve Jigsaw Puzzles with JPDVT on all .JPEG images
in a specified directory. Results (images + logs) are saved in user-defined paths.
"""

import os
import time
import glob
import logging

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from torchvision import transforms
from torchvision.utils import save_image
from einops import rearrange
from sklearn.metrics import pairwise_distances

# Internal imports (ensure these are in the same directory or PYTHONPATH)
from models import DiT_models, get_2d_sincos_pos_embed
from diffusion import create_diffusion

###############################################################################
#                               CONFIGURATIONS
###############################################################################
# Paths
DATA_DIR = "/cluster/home/muhamhz/data/imagenet/test"      # Where test images live
RESULTS_BASE_DIR = "/cluster/home/muhamhz/JPDVT/image_model/inference"
LOGS_DIR = "/cluster/home/muhamhz/JPDVT/image_model/logs"

# Model / Inference params
MODEL_NAME = "JPDVT"
CHECKPOINT_PATH = "/cluster/home/muhamhz/JPDVT/image_model/results/009-imagenet-JPDVT-crop/checkpoints/2850000.pt"
IMAGE_SIZE = 192              # e.g., 192 or 288
GRID_SIZE = 3                 # e.g., 3 => 3x3 puzzle
SEED = 0
NUM_SAMPLING_STEPS = 250

# System / Misc
BATCH_NORM_TRAIN_MODE = True  # Because batchnorm doesn't always behave with batch size=1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png"]  # Filenames to process

###############################################################################
#                               HELPER FUNCTIONS
###############################################################################
def setup_logging():
    """Set up two loggers: one for normal logs, one for errors."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Main log
    log_file = os.path.join(LOGS_DIR, "inference_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )
    
    # Error log for images that fail
    error_file = os.path.join(LOGS_DIR, "inference_errors.txt")
    err_logger = logging.getLogger("error_logger")
    err_handler = logging.FileHandler(error_file, mode='a')
    err_handler.setLevel(logging.ERROR)
    err_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    err_handler.setFormatter(err_formatter)
    err_logger.addHandler(err_handler)
    err_logger.setLevel(logging.ERROR)
    
    return err_logger

def safe_image_save(tensor_img, out_path, nrow=1, normalize=True):
    """Safely save a tensor image using `torchvision.save_image`."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_image(tensor_img, out_path, nrow=nrow, normalize=normalize)

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM:
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
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
    """
    Greedy algorithm to find the permutation order based on the distance matrix.
    """
    sort_list = []
    tmp = np.copy(distance_matrix)
    for _ in range(tmp.shape[1]):
        order = tmp[:, 0].argmin()
        sort_list.append(order)
        tmp = tmp[:, 1:]
        # effectively remove that row so it's not chosen again
        tmp[order, :] = 1e9  
    return sort_list

def load_single_image(image_path, transform=None):
    """
    Loads a single image and applies the transform. Returns a tensor: (1, C, H, W).
    """
    pil_img = Image.open(image_path).convert("RGB")
    if transform is not None:
        tensor_img = transform(pil_img)
    else:
        # If no transform is specified, just do a basic to-tensor
        tensor_img = transforms.ToTensor()(pil_img)
    return tensor_img.unsqueeze(0)  # add batch dimension

###############################################################################
#                               MAIN INFERENCE LOGIC
###############################################################################
def main():
    # Setup logs
    err_logger = setup_logging()
    logging.info("============================================")
    logging.info("Starting Jigsaw Puzzle Inference Script")
    
    # Seed
    torch.manual_seed(SEED)
    torch.set_grad_enabled(False)
    
    # Prepare transforms
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    
    # Load model
    logging.info(f"Loading model [{MODEL_NAME}] from checkpoint: {CHECKPOINT_PATH}")
    model = DiT_models[MODEL_NAME](input_size=IMAGE_SIZE).to(DEVICE)
    
    state_dict = torch.load(CHECKPOINT_PATH, weights_only=False)
    model_state_dict = state_dict['model']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in model_state_dict.items() if k in model_dict}
    model.load_state_dict(pretrained_dict, strict=False)
    
    if BATCH_NORM_TRAIN_MODE:
        model.train()
    
    # Create diffusion
    diffusion = create_diffusion(str(NUM_SAMPLING_STEPS))
    
    # Prepare time embedding
    time_emb = torch.tensor(get_2d_sincos_pos_embed(8, GRID_SIZE)).unsqueeze(0).float().to(DEVICE)
    time_emb_noise = torch.tensor(get_2d_sincos_pos_embed(8, 12)).unsqueeze(0).float().to(DEVICE)
    time_emb_noise = torch.randn_like(time_emb_noise).repeat(1, 1, 1)
    
    logging.info("Model and diffusion initialized.")
    logging.info(f"Reading images from: {DATA_DIR}")
    
    # Get list of all valid images
    image_paths = []
    for ext in ALLOWED_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(DATA_DIR, f"*{ext}")))
    image_paths = sorted(image_paths)
    
    logging.info(f"Found {len(image_paths)} images to process.")
    
    # Track accuracy
    correct_count = 0
    total_count = 0
    start_time = time.time()
    
    for idx, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        base_name, _ = os.path.splitext(filename)
        
        # For printing progress
        logging.info(f"Processing {idx+1}/{len(image_paths)}: {filename}")
        
        try:
            t0 = time.time()
            # Load image
            x = load_single_image(img_path, transform=transform).to(DEVICE)
            
            # ========== Original (for saving) ==========
            # We'll save the original with un-normalized color
            original_unnorm = x * 0.5 + 0.5
            
            # ========== Scramble the puzzle ==========
            # Create random permutation
            indices = np.random.permutation(GRID_SIZE * GRID_SIZE)
            
            # Split into patches
            x_patches = rearrange(x, 'b c (g1 h1) (g2 w1) -> b c (g1 g2) h1 w1',
                                  g1=GRID_SIZE, g2=GRID_SIZE,
                                  h1=IMAGE_SIZE//GRID_SIZE, w1=IMAGE_SIZE//GRID_SIZE)
            
            # Permute
            x_patches = x_patches[:, :, indices, :, :]
            
            # Reassemble scrambled image
            x_scrambled = rearrange(x_patches, 'b c (p1 p2) h1 w1 -> b c (p1 h1) (p2 w1)',
                                    p1=GRID_SIZE, p2=GRID_SIZE,
                                    h1=IMAGE_SIZE//GRID_SIZE, w1=IMAGE_SIZE//GRID_SIZE)
            
            # ========== Run diffusion model ==========
            samples = diffusion.p_sample_loop(
                model.forward,
                x_scrambled,
                time_emb_noise.shape,
                time_emb_noise,
                clip_denoised=False,
                model_kwargs=None,
                progress=False,
                device=DEVICE
            )
            
            # We only have 1 sample here
            sample = samples[0]
            
            # ========== Predict permutation ==========
            # Reshape sample for each patch
            sample_patch_dim = IMAGE_SIZE // (16 * GRID_SIZE)  # from original logic
            sample = rearrange(sample, '(p1 h1 p2 w1) d -> (p1 p2) (h1 w1) d',
                               p1=GRID_SIZE, p2=GRID_SIZE, h1=sample_patch_dim, w1=sample_patch_dim)
            # Mean across spatial dimension
            sample = sample.mean(1)
            
            # Compare with time_emb
            dist = pairwise_distances(sample.cpu().numpy(), time_emb[0].cpu().numpy(), metric='manhattan')
            order = find_permutation(dist)
            pred = np.asarray(order).argsort()
            
            # Evaluate correctness
            is_correct = int((pred == indices).all())
            correct_count += is_correct
            total_count += 1
            
            # Reconstruct final puzzle using predicted order
            scrambled_patches_list = [x_patches[0, :, i, :, :] for i in range(GRID_SIZE * GRID_SIZE)]
            reconstructed_patches = [None] * (GRID_SIZE * GRID_SIZE)
            for i, pos in enumerate(pred):
                reconstructed_patches[pos] = scrambled_patches_list[i]
            
            grid_reconstructed = torch.stack(reconstructed_patches)
            
            # ========== Save results ==========
            # Make the directory for this grid size
            out_dir = os.path.join(RESULTS_BASE_DIR, str(GRID_SIZE))
            os.makedirs(out_dir, exist_ok=True)
            
            # Original
            out_original = os.path.join(out_dir, f"{base_name}_original.png")
            safe_image_save(original_unnorm[0], out_original, nrow=1, normalize=False)
            
            # Scrambled
            scrambled_unnorm = x_scrambled * 0.5 + 0.5
            out_scrambled = os.path.join(out_dir, f"{base_name}_scrambled.png")
            safe_image_save(scrambled_unnorm[0], out_scrambled, nrow=1, normalize=False)
            
            # Reconstructed
            # We'll normalize to match the original puzzle patches approach
            out_reconstructed = os.path.join(
                out_dir, f"{base_name}_reconstructed_acc={float(is_correct):.1f}.png"
            )
            safe_image_save(grid_reconstructed, out_reconstructed,
                            nrow=GRID_SIZE, normalize=True)
            
            elapsed = time.time() - t0
            logging.info(f"   Finished {filename} | Correct? {bool(is_correct)} | Time: {elapsed:.2f}s")
        
        except Exception as e:
            # Log and continue
            err_logger.error(f"Failed on image {filename}: {str(e)}")
            logging.error(f"Skipping {filename} due to error.")
            continue
    
    # ========== Final Stats ==========
    overall_accuracy = (correct_count / total_count) if total_count > 0 else 0.0
    total_time = time.time() - start_time
    logging.info("============================================")
    logging.info(f"Done. Processed {total_count} images. Correct = {correct_count}.")
    logging.info(f"Final Accuracy: {overall_accuracy:.4f}")
    logging.info(f"Total inference time: {total_time:.2f}s")
    logging.info("============================================")

###############################################################################
#                                   LAUNCH
###############################################################################
if __name__ == "__main__":
    main()
