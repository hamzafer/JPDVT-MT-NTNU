#!/usr/bin/env python3
"""
Inference script to solve Jigsaw Puzzles with JPDVT on all .JPEG images
in a specified directory. Results (images + logs) are saved in user-defined paths.

Now includes:
- Puzzle accuracy (entire puzzle correct or not)
- Patch accuracy (fraction of individual patches correct)
- Resume functionality via a CSV log (inference_progress.csv) so the script can
  pick up where it left off if interrupted.

with masks
2025-05-21 19:55:21,705 [INFO] OVERALL RESULTS:
2025-05-21 19:55:21,706 [INFO] Downsampled1: Puzzle Acc = 0.0926, Patch Acc = 0.2798, Images = 54
2025-05-21 19:55:21,706 [INFO] Downsampled2: Puzzle Acc = 0.0278, Patch Acc = 0.2562, Images = 36
"""

import os
import time
import glob
import logging
import csv

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
DATA_DIR = "/cluster/home/muhamhz/data/DataTexRecSSH"  # Parent directory containing both subfolders
RESULTS_BASE_DIR = "/cluster/home/muhamhz/JPDVT/image_model/inference/DataTexRecSSH"
LOGS_DIR = "/cluster/home/muhamhz/JPDVT/image_model/logs/DataTexRecSSH"

# Model / Inference params
MODEL_NAME = "JPDVT"
CHECKPOINT_PATH = "/cluster/home/muhamhz/JPDVT/image_model/models/3x3_Full/2850000.pt"
IMAGE_SIZE = 192              # e.g., 192 or 288
GRID_SIZE = 3                 # e.g., 3 => 3x3 puzzle
SEED = 0
NUM_SAMPLING_STEPS = 250

# System / Misc
BATCH_NORM_TRAIN_MODE = True  # Because batchnorm doesn't always behave with batch size=1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Extensions to search
ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPEG"]

# Name for progress CSV (where we record results per image for resume)
PROGRESS_CSV = "inference_progress.csv"

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

###############################################################################
#                               MAIN INFERENCE LOGIC
###############################################################################
def main():
    # Setup logs
    err_logger = setup_logging()
    logging.info("============================================")
    logging.info("Starting Jigsaw Puzzle Inference Script with Resume")
    
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
    
    # Process both directories
    directories = ["Downsampled1", "Downsampled2"]
    overall_results = {}
    
    for directory in directories:
        dir_path = os.path.join(DATA_DIR, directory)
        logging.info(f"Reading images from: {dir_path}")
        
        # Build list of all valid images
        image_paths = []
        for ext in ALLOWED_EXTENSIONS:
            pattern = os.path.join(dir_path, f'*{ext}')
            image_paths.extend(glob.glob(pattern))
        image_paths = sorted(image_paths)
        
        logging.info(f"Found {len(image_paths)} images in {directory}.")
        
        # Resume logic: load existing progress, if any
        progress_csv_path = os.path.join(LOGS_DIR, f"{directory}_{PROGRESS_CSV}")
        processed_set, puzzle_correct_count, patch_correct_sum, total_count = load_progress_csv(progress_csv_path)
        
        logging.info(f"Resume info: {len(processed_set)} images already processed.")
        if total_count > 0:
            logging.info(
                f"   So far: puzzleAcc = {puzzle_correct_count/total_count:.2f}, "
                f"patchAcc = {patch_correct_sum/(total_count*GRID_SIZE*GRID_SIZE):.2f}"
            )
        
        start_time = time.time()
        # Process each image, skipping those done
        for idx, img_path in enumerate(image_paths):
            filename = os.path.basename(img_path)
            if filename in processed_set:
                # Already done => skip
                continue
            
            logging.info(f"Processing {total_count+1}/{len(image_paths)}: {filename}")
            
            try:
                t0 = time.time()
                # Load image
                x = load_single_image(img_path, transform=transform).to(DEVICE)
                
                # ========== Original (for saving) ==========
                original_unnorm = x * 0.5 + 0.5   # unnormalize for saving
                
                # ========== Scramble the puzzle ==========
                indices = np.random.permutation(GRID_SIZE * GRID_SIZE)
                x_patches = rearrange(
                    x, 'b c (g1 h1) (g2 w1) -> b c (g1 g2) h1 w1',
                    g1=GRID_SIZE, g2=GRID_SIZE,
                    h1=IMAGE_SIZE//GRID_SIZE, w1=IMAGE_SIZE//GRID_SIZE
                )
                x_patches = x_patches[:, :, indices, :, :]  # Permute
                # Reassemble scrambled image
                x_scrambled = rearrange(
                    x_patches, 'b c (p1 p2) h1 w1 -> b c (p1 h1) (p2 w1)',
                    p1=GRID_SIZE, p2=GRID_SIZE,
                    h1=IMAGE_SIZE//GRID_SIZE, w1=IMAGE_SIZE//GRID_SIZE
                )
                
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
                
                sample = samples[0]  # single batch item
                
                # ========== Predict permutation ==========
                sample_patch_dim = IMAGE_SIZE // (16 * GRID_SIZE)
                sample = rearrange(
                    sample, '(p1 h1 p2 w1) d -> (p1 p2) (h1 w1) d',
                    p1=GRID_SIZE, p2=GRID_SIZE,
                    h1=sample_patch_dim, w1=sample_patch_dim
                )
                sample = sample.mean(1)  # average across spatial dimension
                
                # Compare with time_emb
                dist = pairwise_distances(sample.cpu().numpy(), time_emb[0].cpu().numpy(), metric='manhattan')
                order = find_permutation(dist)
                pred = np.asarray(order).argsort()
                
                # Puzzle-level correctness (1 if entire puzzle is correct)
                puzzle_correct = int((pred == indices).all())
                
                # Patch-level correctness (how many positions match?)
                patch_matches = (pred == indices).sum()  # e.g. 0..9 for 3x3
                total_patches_for_img = GRID_SIZE * GRID_SIZE
                patch_accuracy_for_img = patch_matches / total_patches_for_img
                
                # Update counters
                puzzle_correct_count += puzzle_correct
                patch_correct_sum += patch_matches
                total_count += 1
                
                # Reconstruct final puzzle
                scrambled_patches_list = [x_patches[0, :, i, :, :] for i in range(total_patches_for_img)]
                reconstructed_patches = [None] * total_patches_for_img
                for i, pos in enumerate(pred):
                    reconstructed_patches[pos] = scrambled_patches_list[i]
                grid_reconstructed = torch.stack(reconstructed_patches)
                
                # ========== Save results ==========
                out_dir = os.path.join(RESULTS_BASE_DIR, directory, f"Grid{GRID_SIZE}")
                os.makedirs(out_dir, exist_ok=True)
                
                out_original = os.path.join(out_dir, f"{os.path.splitext(filename)[0]}_original.png")
                safe_image_save(original_unnorm[0], out_original, nrow=1, normalize=False)
                
                scrambled_unnorm = x_scrambled * 0.5 + 0.5
                out_scrambled = os.path.join(out_dir, f"{os.path.splitext(filename)[0]}_random.png")
                safe_image_save(scrambled_unnorm[0], out_scrambled, nrow=1, normalize=False)
                
                # Include puzzle correctness in the output filename
                out_reconstructed = os.path.join(
                    out_dir,
                    f"{os.path.splitext(filename)[0]}_reconstructed_pAcc={puzzle_correct}_patchAcc={patch_accuracy_for_img:.2f}.png"
                )
                safe_image_save(grid_reconstructed, out_reconstructed, nrow=GRID_SIZE, normalize=True)
                
                elapsed = time.time() - t0
                
                # Running metrics so far
                puzzle_accuracy_so_far = puzzle_correct_count / total_count
                patch_accuracy_so_far = patch_correct_sum / (total_count * total_patches_for_img)
                
                logging.info(
                    f"   PuzzleAcc={puzzle_correct} PatchAcc={patch_accuracy_for_img:.2f} "
                    f"(Running puzzleAcc={puzzle_accuracy_so_far:.2f}, patchAcc={patch_accuracy_so_far:.2f}) "
                    f"| Time={elapsed:.2f}s"
                )
                
                # Append to progress CSV
                append_progress_csv(
                    progress_csv_path,
                    filename,
                    puzzle_correct,
                    patch_matches,
                    elapsed
                )
            
            except Exception as e:
                err_logger.error(f"Failed on image {filename}: {str(e)}")
                logging.error(f"Skipping {filename} due to error.")
                continue
        
        # ========== Directory Stats ==========
        total_time = time.time() - start_time
        puzzle_accuracy = (puzzle_correct_count / total_count) if total_count > 0 else 0.0
        patch_accuracy = (patch_correct_sum / (total_count * GRID_SIZE * GRID_SIZE)) if total_count > 0 else 0.0
        
        # Store results for this directory
        overall_results[directory] = {
            "puzzle_accuracy": puzzle_accuracy,
            "patch_accuracy": patch_accuracy,
            "total_count": total_count
        }
        
        logging.info("============================================")
        logging.info(f"Done with {directory}. Processed {total_count} images (including resumed ones).")
        logging.info(f"Puzzle Accuracy: {puzzle_accuracy:.4f}")
        logging.info(f"Patch Accuracy: {patch_accuracy:.4f}")
        logging.info(f"Total inference time: {total_time:.2f}s")
        logging.info("============================================")

    # ========== Overall Stats ==========
    logging.info("============================================")
    logging.info("OVERALL RESULTS:")
    for directory, results in overall_results.items():
        logging.info(f"{directory}: Puzzle Acc = {results['puzzle_accuracy']:.4f}, " 
                    f"Patch Acc = {results['patch_accuracy']:.4f}, "
                    f"Images = {results['total_count']}")
    logging.info("============================================")

###############################################################################
#                                   LAUNCH
###############################################################################
if __name__ == "__main__":
    main()
