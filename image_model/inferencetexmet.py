#!/usr/bin/env python3
"""
Inference script to solve Jigsaw Puzzles with JPDVT on all .JPEG images
in a specified directory. Results (images + logs) are saved in user-defined paths.

Now includes:
- Puzzle accuracy (entire puzzle correct or not)
- Patch accuracy (fraction of individual patches correct)
- Resume functionality via a CSV log (inference_progress.csv) so the script can
  pick up where it left off if interrupted.
"""

import os
import time
import glob
import logging
import csv
import argparse

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
# DATA_DIR = "/cluster/home/muhamhz/data/inpainting/gt_images/image"
# DATA_DIR = "/cluster/home/muhamhz/data/inpainting/exp1_irregular_masking/masked"
# DATA_DIR = "/cluster/home/muhamhz/data/inpainting/exp1_irregular_masking/results/imgs"
# DATA_DIR = "/cluster/home/muhamhz/data/inpainting/exp2_regular_masking/masked"
DATA_DIR = "/cluster/home/muhamhz/data/inpainting/exp2_regular_masking/results/imgs"

# Name for progress CSV (where we record results per image for resume)
PROGRESS_CSV = "dsfaffdshfgfghfmbghdsgfdgfgdsfsgfgfdgdjhjdfasdf.csv"

RESULTS_BASE_DIR = "/cluster/home/muhamhz/JPDVT/image_model/inference/inpainting"

# Create descriptive logs directory based on model parameters
MODEL_NAME = "JPDVT"
# CHECKPOINT_PATH = "/cluster/home/muhamhz/JPDVT/image_model/results/006-texmet-JPDVT-crop/checkpoints/final_0077500.pt"
CHECKPOINT_PATH = "/cluster/home/muhamhz/JPDVT/image_model/models/3x3_Full/2850000.pt"
# CHECKPOINT_PATH = "/cluster/home/muhamhz/JPDVT/image_model/results_finetune/003-texmet-JPDVT-crop/checkpoints/final_2927500.pt"
# IMAGE_SIZE = 288              # e.g., 192 or 288
IMAGE_SIZE = 192              # e.g., 192 or 288
GRID_SIZE = 3                 # e.g., 3 => 3x3 puzzle
SEED = 42
NUM_SAMPLING_STEPS = 250

# Extract checkpoint identifier for logs directory
checkpoint_name = os.path.basename(CHECKPOINT_PATH).replace('.pt', '')
LOGS_DIR = f"/cluster/home/muhamhz/JPDVT/image_model/logs/{MODEL_NAME}_img{IMAGE_SIZE}_grid{GRID_SIZE}_steps{NUM_SAMPLING_STEPS}_seed{SEED}_{checkpoint_name}"

# System / Misc
BATCH_NORM_TRAIN_MODE = True  # Because batchnorm doesn't always behave with batch size=1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Extensions to search
ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPEG"]

###############################################################################
#                               HELPER FUNCTIONS
###############################################################################
def setup_logging():
    """Set up two loggers: one for normal logs, one for errors."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Main log file
    log_file = os.path.join(LOGS_DIR, "inference_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )
    
    # Separate error log
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
    """Safely save a tensor image using `torchvision.utils.save_image`."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_image(tensor_img, out_path, nrow=nrow, normalize=normalize)

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
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

def load_progress_csv(csv_path):
    """
    If progress CSV exists, load it and return:
    - processed_set: set of all filenames processed
    - puzzle_correct_count, patch_correct_sum, total_count
    """
    processed_set = set()
    puzzle_correct_count = 0
    patch_correct_sum = 0
    total_count = 0
    
    if not os.path.exists(csv_path):
        return processed_set, puzzle_correct_count, patch_correct_sum, total_count
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            puzzle_correct = int(row["puzzle_correct"])
            patch_matches = int(row["patch_matches"])
            processed_set.add(filename)
            puzzle_correct_count += puzzle_correct
            patch_correct_sum += patch_matches
            total_count += 1
    
    return processed_set, puzzle_correct_count, patch_correct_sum, total_count

def append_progress_csv(csv_path, filename, puzzle_correct, patch_matches, elapsed):
    """
    Append a single line to the progress CSV for the processed image.
    """
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        fieldnames = ["filename", "puzzle_correct", "patch_matches", "time_s"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "filename": filename,
            "puzzle_correct": puzzle_correct,
            "patch_matches": patch_matches,
            "time_s": f"{elapsed:.2f}"
        })

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for parallel processing')
    return parser.parse_args()

def load_batch_images(image_paths, transform, device):
    """Load a batch of images at once"""
    batch_tensors = []
    for img_path in image_paths:
        pil_img = Image.open(img_path).convert("RGB")
        tensor_img = transform(pil_img)
        batch_tensors.append(tensor_img)
    
    # Stack into batch: (batch_size, C, H, W)
    batch = torch.stack(batch_tensors).to(device)
    return batch

###############################################################################
#                               MAIN INFERENCE LOGIC
###############################################################################
def main():
    args = parse_args()
    
    # Setup logs
    err_logger = setup_logging()
    logging.info("============================================")
    logging.info("Starting MULTI-GPU Jigsaw Puzzle Inference Script")
    
    # Check GPU availability
    num_gpus = torch.cuda.device_count()
    logging.info(f"Found {num_gpus} GPUs available")
    
    # Seed EVERYTHING for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)  # ADD THIS LINE!
    torch.set_grad_enabled(False)
    
    logging.info(f"Set random seed to {SEED} for reproducibility")
    
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
    
    # **MULTI-GPU MAGIC: Wrap model with DataParallel**
    if num_gpus > 1:
        logging.info(f"Using DataParallel across {num_gpus} GPUs!")
        model = torch.nn.DataParallel(model)
        BATCH_SIZE = args.batch_size * num_gpus  # Scale batch size
    else:
        BATCH_SIZE = args.batch_size
    
    if BATCH_NORM_TRAIN_MODE:
        model.train()
    
    # Create diffusion
    diffusion = create_diffusion(str(NUM_SAMPLING_STEPS))
    
    # Prepare time embedding (expand for batch processing)
    time_emb = torch.tensor(get_2d_sincos_pos_embed(8, GRID_SIZE)).unsqueeze(0).float().to(DEVICE)
    PATCHES_PER_SIDE = IMAGE_SIZE // 16
    time_emb_noise_single = torch.tensor(get_2d_sincos_pos_embed(8, PATCHES_PER_SIDE)).unsqueeze(0).float().to(DEVICE)
    time_emb_noise_single = torch.randn_like(time_emb_noise_single)
    
    logging.info("Model and diffusion initialized.")
    logging.info(f"Reading images from: {DATA_DIR}")
    
    # Get all images
    image_paths = []
    for ext in ALLOWED_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(DATA_DIR, f"*{ext}")))
        if ext.lower() != ext:
            image_paths.extend(glob.glob(os.path.join(DATA_DIR, f"*{ext.lower()}")))

    image_paths = sorted(list(set(image_paths)))
    logging.info(f"Found {len(image_paths)} images in {DATA_DIR}.")
    logging.info(f"Processing in batches of {BATCH_SIZE}")
    
    # Resume logic
    progress_csv_path = os.path.join(LOGS_DIR, PROGRESS_CSV)
    processed_set, puzzle_correct_count, patch_correct_sum, total_count = load_progress_csv(progress_csv_path)
    
    # Filter out already processed images
    remaining_paths = [path for path in image_paths if os.path.basename(path) not in processed_set]
    logging.info(f"Resume info: {len(processed_set)} images already processed. {len(remaining_paths)} remaining.")
    
    start_time = time.time()
    
    # **BATCH PROCESSING WITH MULTI-GPU**
    for batch_start in range(0, len(remaining_paths), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(remaining_paths))
        batch_paths = remaining_paths[batch_start:batch_end]
        actual_batch_size = len(batch_paths)
        
        if actual_batch_size == 0:
            break
            
        logging.info(f"Processing batch {batch_start//BATCH_SIZE + 1}: images {batch_start+1}-{batch_end}")
        
        try:
            t0 = time.time()
            
            # Load batch of images
            x_batch = load_batch_images(batch_paths, transform, DEVICE)
            
            # Expand time embeddings for batch
            time_emb_noise_batch = time_emb_noise_single.repeat(actual_batch_size, 1, 1)
            
            # Process entire batch at once
            batch_results = []
            
            for i in range(actual_batch_size):
                x = x_batch[i:i+1]  # Single image from batch
                
                # Scramble
                indices = np.random.permutation(GRID_SIZE * GRID_SIZE)
                x_patches = rearrange(
                    x, 'b c (g1 h1) (g2 w1) -> b c (g1 g2) h1 w1',
                    g1=GRID_SIZE, g2=GRID_SIZE,
                    h1=IMAGE_SIZE//GRID_SIZE, w1=IMAGE_SIZE//GRID_SIZE
                )
                x_patches = x_patches[:, :, indices, :, :]
                x_scrambled = rearrange(
                    x_patches, 'b c (p1 p2) h1 w1 -> b c (p1 h1) (p2 w1)',
                    p1=GRID_SIZE, p2=GRID_SIZE,
                    h1=IMAGE_SIZE//GRID_SIZE, w1=IMAGE_SIZE//GRID_SIZE
                )
                
                batch_results.append((x, x_scrambled, x_patches, indices))
            
            # **MULTI-GPU DIFFUSION: Process all scrambled images at once**
            all_scrambled = torch.cat([result[1] for result in batch_results], dim=0)
            
            # This runs on ALL GPUs automatically!
            all_samples = diffusion.p_sample_loop(
                model,  # DataParallel model uses all GPUs
                all_scrambled,
                time_emb_noise_batch.shape,
                time_emb_noise_batch,
                clip_denoised=False,
                model_kwargs=None,
                progress=False,
                device=DEVICE
            )
            
            # Process results for each image in batch
            for i, (x, x_scrambled, x_patches, indices) in enumerate(batch_results):
                sample = all_samples[i]
                
                # Predict permutation
                sample_patch_dim = IMAGE_SIZE // (16 * GRID_SIZE)
                sample_rearranged = rearrange(
                    sample, '(p1 h1 p2 w1) d -> (p1 p2) (h1 w1) d',
                    p1=GRID_SIZE, p2=GRID_SIZE,
                    h1=sample_patch_dim, w1=sample_patch_dim
                )
                sample = sample_rearranged.mean(1)
                
                dist = pairwise_distances(sample.cpu().numpy(), time_emb[0].cpu().numpy(), metric='manhattan')
                order = find_permutation(dist)
                pred = np.asarray(order).argsort()
                
                # Calculate metrics
                puzzle_correct = int((pred == indices).all())
                patch_matches = (pred == indices).sum()
                patch_accuracy_for_img = patch_matches / (GRID_SIZE * GRID_SIZE)
                
                # Update counters
                puzzle_correct_count += puzzle_correct
                patch_correct_sum += patch_matches
                total_count += 1
                
                # Save results
                filename = os.path.basename(batch_paths[i])
                
                # Save images (simplified for speed)
                out_dir = os.path.join(RESULTS_BASE_DIR, f"Grid{GRID_SIZE}")
                os.makedirs(out_dir, exist_ok=True)
                
                # Reconstruct and save only the final result
                scrambled_patches_list = [x_patches[0, :, j, :, :] for j in range(GRID_SIZE * GRID_SIZE)]
                reconstructed_patches = [None] * (GRID_SIZE * GRID_SIZE)
                for j, pos in enumerate(pred):
                    reconstructed_patches[pos] = scrambled_patches_list[j]
                grid_reconstructed = torch.stack(reconstructed_patches)
                
                out_reconstructed = os.path.join(
                    out_dir,
                    f"{os.path.splitext(filename)[0]}_result_pAcc={puzzle_correct}_patchAcc={patch_accuracy_for_img:.2f}.png"
                )
                safe_image_save(grid_reconstructed, out_reconstructed, nrow=GRID_SIZE, normalize=True)
                
                # Log progress
                elapsed = time.time() - t0
                puzzle_accuracy_so_far = puzzle_correct_count / total_count
                patch_accuracy_so_far = patch_correct_sum / (GRID_SIZE * GRID_SIZE * total_count)
                
                # Append to CSV
                append_progress_csv(progress_csv_path, filename, puzzle_correct, patch_matches, elapsed/actual_batch_size)
            
            batch_time = time.time() - t0
            logging.info(f"Batch completed in {batch_time:.2f}s ({batch_time/actual_batch_size:.2f}s per image)")
            logging.info(f"Running accuracy: Puzzle={puzzle_correct_count/total_count:.3f}, Patch={patch_correct_sum/(GRID_SIZE*GRID_SIZE*total_count):.3f}")
            
        except Exception as e:
            err_logger.error(f"Failed on batch starting at {batch_start}: {str(e)}")
            logging.error(f"Skipping batch due to error.")
            continue
    
    # Final stats
    total_time = time.time() - start_time
    puzzle_accuracy = (puzzle_correct_count / total_count) if total_count > 0 else 0.0
    patch_accuracy = (patch_correct_sum / (total_count * GRID_SIZE * GRID_SIZE)) if total_count > 0 else 0.0
    
    logging.info("============================================")
    logging.info(f"MULTI-GPU PROCESSING COMPLETE!")
    logging.info(f"Processed {total_count} images using {num_gpus} GPUs")
    logging.info(f"Final Puzzle Accuracy: {puzzle_accuracy:.4f}")
    logging.info(f"Final Patch Accuracy: {patch_accuracy:.4f}")
    logging.info(f"Total time: {total_time:.2f}s ({total_time/total_count:.2f}s per image)")
    logging.info(f"Speedup: ~{num_gpus}x faster than single GPU!")
    logging.info("============================================")

###############################################################################
#                                   LAUNCH
###############################################################################
if __name__ == "__main__":
    main()
