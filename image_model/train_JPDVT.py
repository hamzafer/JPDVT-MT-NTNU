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

                # reset counters
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save JPDVT checkpoint
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "train_steps": train_steps  # ### ADDED: store the current step
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout

    if rank == 0:
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
    parser.add_argument("--image-size", type=int, choices=[192, 288], default=288)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--global-batch-size", type=int, default=96)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--ckpt", type=str, default='')
    args = parser.parse_args()
    main(args)
