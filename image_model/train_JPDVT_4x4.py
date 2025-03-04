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

from models import DiT_models
from models import get_2d_sincos_pos_embed
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

from datasets import MET
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
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
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
        model_string_name = args.dataset+"-" + model_string_name + "-crop" if args.crop else args.dataset+"-" + model_string_name 
        model_string_name = model_string_name + "-withmask" if args.add_mask else model_string_name
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # ### CHANGED/REMOVED ### 
    # Remove or relax the old "image_size % 3 == 0" assertion.
    #   Because for 4×4 puzzle, we need multiples of 4 instead.
    # ---------------------------------------------------------
    # OLD:
    # assert args.image_size % 3 == 0, "Image size should be Multiples of 3"
    # 
    # We'll simply remove it or check for 3 or 4:
    if args.image_size not in [192, 256, 288]:
        logger.info("Warning: image_size not in [192, 256, 288]. Proceeding anyway...")

    # Additional checks for special logic:
    # For 3×3 puzzle => image-size=192 or 288 w/ center-cropping
    # For 4×4 puzzle => image-size=256
    # 
    # But let's skip strict forcing in code; just proceed if user sets them properly.

    if args.dataset == 'imagenet' and args.crop:
        logger.info("Will rearrange into puzzle patches using the 'crop' logic.")

    # Create model:
    model = DiT_models[args.model](input_size=args.image_size)

    # Optionally load checkpoint
    if args.ckpt != "":
        ckpt_path = args.ckpt
        print("Loading model from", ckpt_path)
        model_dict = model.state_dict()
        state_dict = torch.load(ckpt_path, weights_only=False)
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model.load_state_dict(pretrained_dict, strict=False)

    # Create EMA model
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)
    diffusion = create_diffusion(timestep_respacing="")
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Basic transforms
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5], inplace=True)
    ])

    # Setup dataset
    if args.dataset == "met":
        dataset = MET(args.data_path, 'train')
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
    logger.info(f"Dataset contains {len(dataset):,} images")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        for x_batch in loader:
            if args.dataset == 'imagenet':
                x_batch, _ = x_batch  # ImageFolder returns (img, label)

            x_batch = x_batch.to(device)
            puzzle_dim = 3

            # ### CHANGED / ADDED ###
            # If we are using --crop with imagenet, we want to do puzzle rearrangement.
            # Hardcode logic for 3×3 or 4×4 puzzle based on image_size:
            if args.dataset == 'imagenet' and args.crop:
                if args.image_size == 192:
                    # 3×3 puzzle
                    # Each full image is 192×192 => 3 patches in each dimension => each patch is 64×64 after center-crop.
                    # So we first shape them into (3×96) => center-crop(64×64).
                    centercrop = transforms.CenterCrop((64, 64))
                    x_resh = rearrange(
                        x_batch, 'b c (p1 h1) (p2 w1) -> b c (p1 p2) h1 w1',
                        p1=3, p2=3, h1=96, w1=96
                    )
                    x_resh = centercrop(x_resh)  # from 96×96 -> 64×64
                    x_batch = rearrange(
                        x_resh, 'b c (p1 p2) h1 w1 -> b c (p1 h1) (p2 w1)',
                        p1=3, p2=3, h1=64, w1=64
                    )

                elif args.image_size == 256:
                    puzzle_dim = 4
                    # 4×4 puzzle
                    # Each full image is 256×256 => 4 patches in each dimension => each patch is ~64×64.
                    # We can replicate the logic from above but with p1=4, p2=4, etc.
                    centercrop = transforms.CenterCrop((64, 64))
                    # Split into 4×4 => each patch is 64+something => let's assume 64×64 is final:
                    # We can do 4×(64+16) = 4×80 = 320, so that would require the image to be 320. 
                    # But we have 256, so each patch is already 64 => no center-crop needed. 
                    # For consistency with the code, let's do a small center-crop (if we want). 
                    # If the entire image is 256×256, each patch is exactly 64×64 => no crop required. 
                    # Let's skip the centercrop to keep it minimal:
                    x_resh = rearrange(
                        x_batch, 'b c (p1 h1) (p2 w1) -> b c (p1 p2) h1 w1',
                        p1=4, p2=4, h1=64, w1=64
                    )
                    # If you REALLY want to center-crop inside each patch, you'd do:
                    # x_resh = centercrop(x_resh)
                    # but that would shrink each patch to 48×48. We'll skip that.
                    x_batch = rearrange(
                        x_resh, 'b c (p1 p2) h1 w1 -> b c (p1 h1) (p2 w1)',
                        p1=4, p2=4, h1=64, w1=64
                    )

            # Build 2D sin-cos embedding (example: 8-dim)
            time_emb = torch.tensor(get_2d_sincos_pos_embed(8, 4)).unsqueeze(0).float().to(device)

            # Random diffusion timestep
            t = torch.randint(0, diffusion.num_timesteps, (x_batch.shape[0],), device=device)
            # You can pass additional kwargs if needed
            loss_dict = diffusion.training_losses(
                model,
                x_batch,
                t,
                time_emb,
                model_kwargs=None,
                block_size=args.image_size // puzzle_dim,
                patch_size=64,
                add_mask=args.add_mask
            )
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            # Logging
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) loss: {avg_loss:.4f}, steps/sec: {steps_per_sec:.2f}")
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()
    logger.info("Done!")
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="JPDVT")
    parser.add_argument("--dataset", type=str, choices=["imagenet", "met"], default="imagenet")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--crop", action='store_true', default=False)
    parser.add_argument("--add-mask", action='store_true', default=False)
    # ### CHANGED ### allow 256 for 4x4:
    parser.add_argument("--image-size", type=int, choices=[192, 256, 288], default=288)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--global-batch-size", type=int, default=96)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--ckpt", type=str, default='')
    args = parser.parse_args()
    main(args)
