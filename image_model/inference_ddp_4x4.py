#!/usr/bin/env python3
"""
Inference script to solve Jigsaw Puzzles with JPDVT on all .JPEG images
in a specified directory. Results (images + logs) are saved in user-defined paths.

Now includes:
- Puzzle accuracy (entire puzzle correct or not)
- Patch accuracy (fraction of individual patches correct)
- Resume functionality via a CSV log (inference_progress.csv) so the script can
  pick up where it left off if interrupted.

ADDED FOR DISTRIBUTED + SINGLE-FILE SAVING:
-------------------------------------------
- Minimal lines to handle multiple GPUs (DDP). Each rank processes a subset of images.
- A tiny helper to combine Original|Scrambled|Reconstructed into ONE file,
  without removing your existing lines that save 3 separate files.
"""

import os
import time
import glob
import logging
import csv

import torch
import torch.distributed as dist  # <--- ADDED for distributed
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
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
DATA_DIR = "/cluster/home/muhamhz/data/imagenet/test"
RESULTS_BASE_DIR = "/cluster/home/muhamhz/JPDVT/image_model/inference"
LOGS_DIR = "/cluster/home/muhamhz/JPDVT/image_model/logs"

# Model / Inference params
MODEL_NAME = "JPDVT-T"
CHECKPOINT_PATH = "/cluster/home/muhamhz/JPDVT/image_model/results/000-imagenet-JPDVT-T-crop/checkpoints/1400000.pt"
IMAGE_SIZE = 256              # e.g., 192 or 288
GRID_SIZE = 4                # e.g., 3 => 3x3 puzzle
SEED = 0
NUM_SAMPLING_STEPS = 250

# System / Misc
BATCH_NORM_TRAIN_MODE = False  # Because batchnorm doesn't always behave with batch size=1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Extensions to search
ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPEG"]

# Name for progress CSV (where we record results per image for resume)
PROGRESS_CSV = "inference_progress.csv"

###############################################################################
#                            ADDED: DISTRIBUTED SETUP
###############################################################################
def init_distributed():
    """
    Initialize PyTorch Distributed if available, else do nothing.
    We'll assume 'nccl' backend for multi-GPU.
    """
    # If you only run "torchrun" or "srun torchrun", dist should be available
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size

###############################################################################
#                          HELPER FUNCTION: COMBINE
###############################################################################
def combine_into_one_image(original, scrambled, reconstructed, grid_size):
    """
    Combine the three images/tensors into ONE side-by-side image.
    This is the single new change for saving all 3 in one file.
    """
    # Convert each to unnormalized PIL
    # original, scrambled are (C,H,W) single images
    # reconstructed is (N, C, H, W) where N=grid_size^2
    # We'll build a wide image: [Original | Scrambled | Reconstructed]

    # 1) Unnormalize the first two
    orig_unnorm = original * 0.5 + 0.5
    orig_unnorm = orig_unnorm.clamp(0,1)
    scr_unnorm = scrambled * 0.5 + 0.5
    scr_unnorm = scr_unnorm.clamp(0,1)

    # 2) Convert them to PIL
    def to_pil(img_tensor):
        arr = (img_tensor*255).permute(1,2,0).cpu().numpy().astype(np.uint8)
        return Image.fromarray(arr)
    pil_orig = to_pil(orig_unnorm)
    pil_scr = to_pil(scr_unnorm)

    # 3) Reconstructed: use make_grid
    rec_grid = torchvision.utils.make_grid(reconstructed, nrow=grid_size, normalize=True)
    pil_rec = transforms.ToPILImage()(rec_grid)

    # 4) Create side-by-side
    w,h = pil_orig.size
    # Ensure consistent height if they differ slightly
    # We'll assume they're all the same for simplicity
    combined_w = w*3
    combined = Image.new("RGB", (combined_w,h), (255,255,255))

    combined.paste(pil_orig, (0,0))
    combined.paste(pil_scr, (w,0))
    combined.paste(pil_rec, (2*w,0))

    # (Optional) draw small text
    draw = ImageDraw.Draw(combined)
    draw.text((5,5),        "Original",      fill=(255,0,0))
    draw.text((w+5,5),      "Scrambled",     fill=(255,0,0))
    draw.text((2*w+5,5),    "Reconstructed", fill=(255,0,0))

    return combined

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
    # ==================== ADDED: DISTRIBUTED INIT ===================
    rank, world_size = 0, 1
    try:
        rank, world_size = init_distributed()  # sets device, etc.
    except:
        pass  # if you run normally (python inference_ddp.py), it won't break
    
    # Setup logs
    err_logger = setup_logging()
    logging.info("============================================")
    logging.info(f"Starting Jigsaw Puzzle Inference Script with Resume (rank={rank}, world_size={world_size})")
    
    # Seed
    torch.manual_seed(SEED + rank)  # different seed per rank
    torch.set_grad_enabled(False)
    
    # Prepare transforms
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    
    # Load model
    if rank == 0:
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
    time_emb_noise = torch.tensor(get_2d_sincos_pos_embed(8, 4)).unsqueeze(0).float().to(DEVICE)
    time_emb_noise = torch.randn_like(time_emb_noise).repeat(1, 1, 1)
    
    if rank == 0:
        logging.info("Model and diffusion initialized.")
        logging.info(f"Reading images from: {DATA_DIR}")
    
    # Build list of all valid images (recursively)
    image_paths = []
    for ext in ALLOWED_EXTENSIONS:
        pattern = os.path.join(DATA_DIR, '**', f'*{ext}')
        image_paths.extend(glob.glob(pattern, recursive=True))
    image_paths = sorted(image_paths)
    
    if rank == 0:
        logging.info(f"Found {len(image_paths)} images total.")
    
    # DISPATCH images among ranks:
    my_image_paths = image_paths[rank::world_size]
    
    # Resume logic: load existing progress, if any
    progress_csv_path = os.path.join(LOGS_DIR, PROGRESS_CSV)
    processed_set, puzzle_correct_count, patch_correct_sum, total_count = load_progress_csv(progress_csv_path)
    
    if rank == 0:
        logging.info(f"Resume info: {len(processed_set)} images already processed.")
        if total_count > 0:
            logging.info(
                f"   So far: puzzleAcc = {puzzle_correct_count/total_count:.2f}, "
                f"patchAcc = {patch_correct_sum/(total_count*GRID_SIZE*GRID_SIZE):.2f}"
            )
    
    # ADDED: Initialize local counters proportionally based on rank
    local_puzzle_correct_count = puzzle_correct_count // world_size
    local_patch_correct_sum = patch_correct_sum // world_size
    local_total_count = total_count // world_size
    
    start_time = time.time()
    # Process each image, skipping those done
    for idx, img_path in enumerate(my_image_paths):
        filename = os.path.basename(img_path)
        if filename in processed_set:
            # Already done => skip
            continue
        
        # For rank logs
        logging.info(f"[Rank={rank}] Processing {local_total_count+1}/{len(my_image_paths)}: {filename}")
        
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
            scrambled_unnorm = x_scrambled * 0.5 + 0.5
            
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
            sample_patch_dim = IMAGE_SIZE // (64 * GRID_SIZE)
            sample = rearrange(
                sample, '(p1 h1 p2 w1) d -> (p1 p2) (h1 w1) d',
                p1=GRID_SIZE, p2=GRID_SIZE,
                h1=sample_patch_dim, w1=sample_patch_dim
            )
            sample = sample.mean(1)  # average across spatial dimension
            
            # Compare with time_emb
            distance_matrix = pairwise_distances(sample.cpu().numpy(), time_emb[0].cpu().numpy(), metric='manhattan')
            order = find_permutation(distance_matrix)
            pred = np.asarray(order).argsort()
            
            # Puzzle-level correctness (1 if entire puzzle is correct)
            puzzle_correct = int((pred == indices).all())
            
            # Patch-level correctness (how many positions match?)
            patch_matches = (pred == indices).sum()  # e.g. 0..9 for 3x3
            total_patches_for_img = GRID_SIZE * GRID_SIZE
            patch_accuracy_for_img = patch_matches / total_patches_for_img
            
            # Update counters
            local_puzzle_correct_count += puzzle_correct
            local_patch_correct_sum += patch_matches
            local_total_count += 1
            
            # Reconstruct final puzzle
            scrambled_patches_list = [x_patches[0, :, i, :, :] for i in range(total_patches_for_img)]
            reconstructed_patches = [None] * total_patches_for_img
            for i, pos in enumerate(pred):
                reconstructed_patches[pos] = scrambled_patches_list[i]
            grid_reconstructed = torch.stack(reconstructed_patches)
            
            # Unnormalize and then reassemble into a single (3, H, W) image
            grid_reconstructed_unnorm = grid_reconstructed * 0.5 + 0.5
            final_reconstructed = rearrange(
                grid_reconstructed_unnorm,
                '(g1 g2) c h w -> c (g1 h) (g2 w)',
                g1=GRID_SIZE, g2=GRID_SIZE
            )
            
            # ========== SAVE A SINGLE COMBINED IMAGE ==========
            # Concatenate horizontally: original | spacer | scrambled | spacer | reconstructed

            # (1) Decide how many pixels of spacing you want
            spacing_width = 10
            # (2) Create a white spacer of shape (3, H, spacing_width)
            #     Here, H = IMAGE_SIZE because original_unnorm, scrambled_unnorm, final_reconstructed all share height.
            white_spacer = torch.ones((3, IMAGE_SIZE, spacing_width), device=DEVICE)

            # (3) Concatenate all: original + spacer + scrambled + spacer + reconstructed
            combined = torch.cat([
                original_unnorm[0],
                white_spacer,            # first spacer
                scrambled_unnorm[0],
                white_spacer,            # second spacer
                final_reconstructed
            ], dim=2)

            # Output name includes puzzle correctness and patch accuracy
            out_dir = os.path.join(RESULTS_BASE_DIR, f"Grid{GRID_SIZE}")
            out_combined = os.path.join(
                out_dir,
                f"{os.path.splitext(filename)[0]}_combined_pAcc={puzzle_correct}_patchAcc={patch_accuracy_for_img:.2f}.png"
            )
            safe_image_save(combined, out_combined, nrow=1, normalize=False)

            elapsed = time.time() - t0
            
            # Running metrics so far (just for rank's local data)
            puzzle_accuracy_so_far = local_puzzle_correct_count / local_total_count
            patch_accuracy_so_far = local_patch_correct_sum / (local_total_count * total_patches_for_img)
            
            logging.info(
                f"[Rank={rank}]   PuzzleAcc={puzzle_correct} PatchAcc={patch_accuracy_for_img:.2f} "
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
    
    # ========== Final Stats (LOCAL) ==========
    local_time = time.time() - start_time
    
    # ========== ALLREDUCE to get global stats =============
    #  We'll sum puzzle_correct, patch_correct, total_count across ranks
    if dist.is_initialized():
        stats_tensor = torch.tensor([local_puzzle_correct_count, local_patch_correct_sum, local_total_count],
                                    dtype=torch.float32, device=DEVICE)
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
        puzzle_correct_count = stats_tensor[0].item()
        patch_correct_sum = stats_tensor[1].item()
        total_count = stats_tensor[2].item()
        # We'll also allreduce max of local_time if you like:
        local_t = torch.tensor([local_time], dtype=torch.float32, device=DEVICE)
        dist.all_reduce(local_t, op=dist.ReduceOp.MAX)
        global_wall_time = local_t[0].item()
    else:
        puzzle_correct_count = local_puzzle_correct_count
        patch_correct_sum = local_patch_correct_sum
        global_wall_time = local_time

    # Only rank=0 prints final stats
    rank0 = True
    if dist.is_initialized() and dist.get_rank() != 0:
        rank0 = False
    
    if rank0:
        puzzle_accuracy = (puzzle_correct_count / total_count) if total_count > 0 else 0.0
        patch_accuracy = (patch_correct_sum / (total_count * GRID_SIZE * GRID_SIZE)) if total_count > 0 else 0.0
        
        logging.info("============================================")
        logging.info(f"Done. Processed {int(total_count)} images (including resumed ones).")
        logging.info(f"Final Puzzle Accuracy: {puzzle_accuracy:.4f}")
        logging.info(f"Final Patch Accuracy: {patch_accuracy:.4f}")
        logging.info(f"Total inference time (wall): {global_wall_time:.2f}s")
        logging.info("============================================")

###############################################################################
#                                   LAUNCH
###############################################################################
if __name__ == "__main__":
    main()
