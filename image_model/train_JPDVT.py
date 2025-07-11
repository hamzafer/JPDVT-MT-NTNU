"""
A minimal training script for JPDVT using PyTorch DDP.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import wandb

from models import DiT_models
from models import get_2d_sincos_pos_embed
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

from datasets import MET, TEXMET
from einops import rearrange

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # If you only want to update params that require_grad, you could do that here.
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


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


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        model_string_name = args.dataset + "-" + model_string_name + "-crop" if args.crop else args.dataset + "-" + model_string_name
        model_string_name = model_string_name + "-withmask" if args.add_mask else model_string_name
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        
        # Create descriptive run name
        run_name_parts = [
            f"exp{experiment_index:03d}",
            args.dataset.upper(),
            args.model,
            f"img{args.image_size}",
            f"bs{args.global_batch_size}",
            f"ep{args.epochs}",
            f"lr{1e-4}".replace(".", ""),
            f"seed{args.global_seed}",
            f"gpu{dist.get_world_size()}"
        ]
        
        # Add optional flags
        if args.crop:
            run_name_parts.append("CROP")
        if args.add_mask:
            run_name_parts.append("MASK")
        if args.ckpt:
            run_name_parts.append("RESUME")
        
        # Add custom tag if provided
        if hasattr(args, 'wandb_tag') and args.wandb_tag:
            run_name_parts.append(args.wandb_tag.upper())
        
        descriptive_run_name = "-".join(run_name_parts)
        
        # Initialize wandb only on rank 0
        if not args.disable_wandb:
            wandb.init(
                project="JPDVT",
                entity="hamzafer3-ntnu",
                name=descriptive_run_name,
                tags=[
                    args.dataset,
                    args.model,
                    f"img_{args.image_size}",
                    f"bs_{args.global_batch_size}",
                    "crop" if args.crop else "no_crop",
                    "mask" if args.add_mask else "no_mask",
                    "resume" if args.ckpt else "fresh",
                    f"gpu_{dist.get_world_size()}"
                ],
                config={
                    "model": args.model,
                    "dataset": args.dataset,
                    "image_size": args.image_size,
                    "epochs": args.epochs,
                    "global_batch_size": args.global_batch_size,
                    "learning_rate": 1e-4,
                    "weight_decay": 0,
                    "global_seed": args.global_seed,
                    "crop": args.crop,
                    "add_mask": args.add_mask,
                    "world_size": dist.get_world_size(),
                    "experiment_dir": experiment_dir,
                    "log_every": args.log_every,
                    "ckpt_every": args.ckpt_every,
                    "num_workers": args.num_workers,
                    "resume_from": args.ckpt if args.ckpt else None,
                    "optimizer": "AdamW",
                    "diffusion_steps": 1000,
                    "grid_size": "3x3",
                    "dataset_size": "18644_images",
                    # System info in config instead of logging
                    "system_info": {
                        "gpu_count": torch.cuda.device_count(),
                    }
                },
                resume="allow" if args.ckpt else None
            )
            
            # Log initial system info as a single entry
            wandb.log({
                "system_gpu_count": torch.cuda.device_count(),
            }, step=0)
            
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 3 == 0, "Image size should be multiples of 3"
    if args.dataset == 'imagenet':
        assert args.image_size == 288 or args.crop, "Use --image-size=192 if you run imagenet with gap"
    model = DiT_models[args.model](
        input_size=args.image_size
    )

    # ### ADDED: We'll track our training steps here if we can load from checkpoint
    train_steps = 0

    ########################################
    # FIRST, wrap the model in DDP or load?
    ########################################
    # Actually, let's do the approach you had: we define the model, then wrap it in DDP:
    # (So "model.module" is valid below).
    # Move to device and wrap in DDP
    model.to(device)
    model = DDP(model, device_ids=[rank])

    # Create the EMA model from the current model's weights:
    # Note: because we only need its parameters, we pass model.module to keep the "raw" underlying network.
    ema = deepcopy(model.module).to(device)
    requires_grad(ema, False)

    # ### CHANGED: If a checkpoint is provided, load it fully (model, ema, opt, train_steps).
    if args.ckpt != "":
        ckpt_path = args.ckpt
        print(f"Rank={rank}: Loading checkpoint from {ckpt_path}")

        # For PyTorch 2.6, ensure weights_only=False so we can load non-tensor objects
        checkpoint = torch.load(ckpt_path, weights_only=False)

        if "model" in checkpoint:
            model.module.load_state_dict(checkpoint["model"], strict=False)
            print(f"Rank={rank}: Loaded 'model' state_dict from checkpoint.")
        else:
            print(f"Rank={rank}: WARNING: 'model' not found in checkpoint.")

        if "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
            print(f"Rank={rank}: Loaded 'ema' state_dict from checkpoint.")
        else:
            print(f"Rank={rank}: WARNING: 'ema' not found in checkpoint.")

        # We'll create the optimizer below, so let's also restore if we have it:
        # (But only rank=0 logs messages to avoid confusion)
        # We'll store "train_steps" if present.
        if "opt" in checkpoint:
            opt_state = checkpoint["opt"]
        else:
            opt_state = None
            print(f"Rank={rank}: WARNING: 'opt' not found in checkpoint.")

        # Attempt to load "train_steps" if present
        if "train_steps" in checkpoint:
            train_steps = checkpoint["train_steps"]
            if rank == 0:
                print(f"Resuming from step={train_steps} for rank0.")
        else:
            if rank == 0:
                print("No 'train_steps' in checkpoint, so we'll start from 0.")

    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    if rank == 0:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    # ### If we had an optimizer state in the checkpoint, load it now
    if args.ckpt != "" and opt_state is not None:
        opt.load_state_dict(opt_state)
        if rank == 0:
            print("Optimizer state loaded from checkpoint.")

    # Setup transforms
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 288)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    # Setup data:
    if args.dataset == "met":
        dataset = MET(args.data_path, 'train')
    elif args.dataset == "texmet":
        dataset = TEXMET(args.data_path, 'train')
    elif args.dataset == "imagenet":
        dataset = ImageFolder(args.data_path, transform=transform)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if rank == 0:
        logger.info(f"Dataset contains {len(dataset):,} images")

    # Prepare models for training:
    # If we're continuing from a checkpoint, we've already loaded model & ema states above.
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()     # EMA model should always be in eval mode

    log_steps = 0
    running_loss = 0
    start_time = time()

    if rank == 0:
        logger.info(f"Training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        if rank == 0:
            logger.info(f"Beginning epoch {epoch}...")

        for x in loader:
            if args.dataset == 'imagenet':
                x, _ = x  # discard class labels
            x = x.to(device)

            if args.dataset == 'imagenet' and args.crop:
                centercrop = transforms.CenterCrop((64, 64))
                patchs = rearrange(x, 'b c (p1 h1) (p2 w1)-> b c (p1 p2) h1 w1', p1=3, p2=3, h1=96, w1=96)
                patchs = centercrop(patchs)
                x = rearrange(patchs, 'b c (p1 p2) h1 w1-> b c (p1 h1) (p2 w1)', p1=3, p2=3, h1=64, w1=64)

            # Set up initial positional embedding
            time_emb = torch.tensor(get_2d_sincos_pos_embed(8, 3)).unsqueeze(0).float().to(device)

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = None

            loss_dict = diffusion.training_losses(
                model,
                x,
                t,
                time_emb,
                model_kwargs,
                block_size=args.image_size // 3,
                patch_size=16,
                add_mask=args.add_mask,
                grid_size=3  # Explicitly set for 3x3
            )
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            # Log every N steps
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)

                # average loss across ranks
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()

                if rank == 0:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, "
                                f"Train Steps/Sec: {steps_per_sec:.2f}")
                    
                    # Log to wandb
                    if not args.disable_wandb:
                        wandb.log({
                            "train_loss": avg_loss,
                            "train_steps_per_sec": steps_per_sec,
                            "epoch": epoch,
                            "step": train_steps
                        })

                # reset counters
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint and validate
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "train_steps": train_steps
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
                    # Run validation
                    logger.info("Running validation...")
                    try:
                        puzzle_acc, patch_acc = validate_model(model, diffusion, args.data_path, device, logger, args)
                        
                        # Log to wandb with more explicit logging
                        if not args.disable_wandb:
                            wandb.log({
                                "checkpoints/saved_at_step": train_steps,
                                "checkpoints/path": checkpoint_path,
                                "validation/puzzle_accuracy": puzzle_acc,
                                "validation/patch_accuracy": patch_acc,
                                "validation/checkpoint_epoch": epoch
                            }, step=train_steps)
                            # Force commit
                            wandb.log({}, commit=True)
                            logger.info(f"Checkpoint validation logged to wandb: puzzle_acc={puzzle_acc:.4f}")
                    
                    except Exception as e:
                        logger.error(f"Checkpoint validation failed: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                
                dist.barrier()

        # Validate every 100 epochs (in addition to checkpoint validation)
        # CHANGED: Also validate after first epoch for testing
        if (epoch > 0 and epoch % 100 == 0) or epoch == 1:
            if rank == 0:
                logger.info(f"Running validation at epoch {epoch}...")
                try:
                    puzzle_acc, patch_acc = validate_model(model, diffusion, args.data_path, device, logger, args)
                    
                    # Force wandb sync
                    if not args.disable_wandb:
                        wandb.log({
                            "validation/puzzle_accuracy": puzzle_acc,
                            "validation/patch_accuracy": patch_acc,
                            "validation/epoch": epoch
                        }, step=train_steps)
                        # Force sync to ensure data is sent
                        wandb.log({}, commit=True)
                        logger.info(f"Logged validation results to wandb: puzzle_acc={puzzle_acc:.4f}, patch_acc={patch_acc:.4f}")
                    
                except Exception as e:
                    logger.error(f"Validation failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

    model.eval()  # important! This disables randomized embedding dropout

    # Save final checkpoint
    if rank == 0:
        final_checkpoint = {
            "model": model.module.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "args": args,
            "train_steps": train_steps
        }
        final_checkpoint_path = f"{checkpoint_dir}/final_{train_steps:07d}.pt"
        torch.save(final_checkpoint, final_checkpoint_path)
        logger.info(f"Saved final checkpoint to {final_checkpoint_path}")
        
        # Run final validation
        logger.info("Running final validation...")
        puzzle_acc, patch_acc = validate_model(model, diffusion, args.data_path, device, logger, args)
        
        if not args.disable_wandb:
            wandb.log({
                "final/puzzle_accuracy": puzzle_acc,
                "final/patch_accuracy": patch_acc,
                "final/checkpoint_path": final_checkpoint_path,
                "training/final_step": train_steps,
                "training/status": "completed"
            })
            wandb.finish()
        
        logger.info("Done!")
    
    cleanup()

@torch.no_grad()
def validate_model(model, diffusion, data_path, device, logger, args, num_samples=100):
    """
    Validate the model on a subset of images and return accuracy metrics.
    """
    from sklearn.metrics import pairwise_distances
    
    model.eval()
    correct_puzzles = 0
    total_patch_matches = 0
    total_patches = 0
    
    # Setup validation dataset - USE VAL SPLIT!
    if args.dataset == "met":
        val_dataset = MET(data_path, 'val')  # Changed to 'val'
    elif args.dataset == "texmet":
        val_dataset = TEXMET(data_path, 'val')  # Changed to 'val'
    elif args.dataset == "imagenet":
        # For ImageNet, use a validation transform without augmentation
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 288)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        # Assuming ImageNet val folder structure
        val_dataset = ImageFolder(data_path.replace('train', 'val'), transform=transform)
    
    # Create a simple loader for validation (no DDP)
    val_indices = torch.randperm(len(val_dataset))[:num_samples].tolist()
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=2)
    
    if logger:
        logger.info(f"Validation dataset: {len(val_dataset)} images, using {num_samples} samples")
    
    # Setup validation parameters
    G = 3  # Grid size
    time_emb = torch.tensor(get_2d_sincos_pos_embed(8, G)).unsqueeze(0).float().to(device)
    PATCHES_PER_SIDE = args.image_size // 16
    time_emb_noise = torch.tensor(get_2d_sincos_pos_embed(8, PATCHES_PER_SIDE)).unsqueeze(0).float().to(device)
    time_emb_noise = torch.randn_like(time_emb_noise).repeat(1, 1, 1)
    
    # Simple validation diffusion with fewer steps
    val_diffusion = create_diffusion("250")
    
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
    
    if logger:
        logger.info("Starting validation inference...")
    
    for i, batch in enumerate(val_loader):
        if args.dataset == 'imagenet':
            x, _ = batch
        else:
            x = batch
        x = x.to(device)
        
        # Create scrambled puzzle
        indices = np.random.permutation(G * G)
        
        # Split into patches and scramble
        x_patches = rearrange(
            x, 'b c (p1 h1) (p2 w1) -> b c (p1 p2) h1 w1', 
            p1=G, p2=G, h1=args.image_size//G, w1=args.image_size//G
        )
        x_patches = x_patches[:, :, indices, :, :]
        x_scrambled = rearrange(
            x_patches, 'b c (p1 p2) h1 w1 -> b c (p1 h1) (p2 w1)', 
            p1=G, p2=G, h1=args.image_size//G, w1=args.image_size//G
        )
        
        try:
            # Run inference
            samples = val_diffusion.p_sample_loop(
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
                sample_patch_dim = args.image_size // (16 * G)
                
                sample = rearrange(
                    sample, '(p1 h1 p2 w1) d -> (p1 p2) (h1 w1) d', 
                    p1=G, p2=G, h1=sample_patch_dim, w1=sample_patch_dim
                )
                
                sample = sample.mean(1)
                
                dist_matrix = pairwise_distances(sample.cpu().numpy(), time_emb[0].cpu().numpy(), metric='manhattan')
                order = find_permutation(dist_matrix)
                pred = np.asarray(order).argsort()
                
                # Calculate metrics
                puzzle_correct = int((pred == indices).all())
                patch_matches = int((pred == indices).sum())
                
                correct_puzzles += puzzle_correct
                total_patch_matches += patch_matches
                total_patches += G * G
                
        except Exception as e:
            if logger:
                logger.warning(f"Validation sample {i} failed: {e}")
            continue
        
        # Progress logging
        if (i + 1) % 20 == 0 and logger:
            logger.info(f"Validation progress: {i+1}/{num_samples} samples processed")
    
    # Calculate accuracies
    puzzle_accuracy = correct_puzzles / num_samples if num_samples > 0 else 0.0
    patch_accuracy = total_patch_matches / total_patches if total_patches > 0 else 0.0
    
    if logger:
        logger.info(f"=== VALIDATION RESULTS ===")
        logger.info(f"Samples processed: {num_samples}")
        logger.info(f"Correct puzzles: {correct_puzzles}")
        logger.info(f"Total patch matches: {total_patch_matches}")
        logger.info(f"Total patches: {total_patches}")
        logger.info(f"Puzzle Accuracy: {puzzle_accuracy:.4f} ({puzzle_accuracy*100:.2f}%)")
        logger.info(f"Patch Accuracy: {patch_accuracy:.4f} ({patch_accuracy*100:.2f}%)")
        logger.info(f"========================")
    
    model.train()
    return puzzle_accuracy, patch_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="JPDVT")
    parser.add_argument("--dataset", type=str, choices=["imagenet", "met", "texmet"], default="imagenet")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--crop", action='store_true', default=False)
    parser.add_argument("--add-mask", action='store_true', default=False)
    parser.add_argument("--image-size", type=int, choices=[192, 288], default=288)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--global-batch-size", type=int, default=96)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--ckpt", type=str, default='')
    parser.add_argument("--disable-wandb", action='store_true', default=False, help="Disable wandb logging")
    parser.add_argument("--wandb-tag", type=str, default='', help="Custom tag for wandb run name")
    args = parser.parse_args()
    main(args)
