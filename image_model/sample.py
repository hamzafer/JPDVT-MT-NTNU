"""
Solve Jigsaw Puzzles with JPDVT
Compatible with 192×192 images, subdivided into 3×3 (64×64) patches.
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models import DiT_models, get_2d_sincos_pos_embed
import argparse
import numpy as np
from torch.utils.data import DataLoader
from datasets import MET
from einops import rearrange
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

def main(args):
    # ------------------------------
    # 1. Basic Setup
    # ------------------------------
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------
    # 2. Load Model
    # ------------------------------
    model = DiT_models[args.model](
        input_size=args.image_size
    ).to(device)

    print("Load model from:", args.ckpt)
    checkpoint = torch.load(args.ckpt, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)

    # Some BN layers can misbehave with batch=1, so the original code uses train() mode
    model.train()

    # Create diffusion
    diffusion = create_diffusion(str(args.num_sampling_steps))

    # ------------------------------
    # 3. Dataset & Transforms
    # ------------------------------
    if args.dataset == "met":
        # MET dataset creates cropped/stitched images
        dataset = MET(args.data_path, 'test')
    elif args.dataset == "imagenet":
        # Transform: resize to args.image_size, then center-crop
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])
        dataset = ImageFolder(args.data_path, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        drop_last=True
    )

    # ------------------------------
    # 4. Prepare Embeddings
    # ------------------------------
    # time_emb is a small static embedding for the puzzle logic
    time_emb = torch.tensor(
        get_2d_sincos_pos_embed(8, 3)
    ).unsqueeze(0).float().to(device)

    # Another random embedding used by the code
    time_emb_noise = torch.tensor(
        get_2d_sincos_pos_embed(8, 18)
    ).unsqueeze(0).float().to(device)
    time_emb_noise = torch.randn_like(time_emb_noise).repeat(1, 1, 1)

    model_kwargs = None

    # A helper for puzzle piece ordering
    def find_permutation(distance_matrix):
        """
        Greedy approach: pick the patch with min distance each time.
        We shift the matrix after each pick.
        """
        sort_list = []
        for _ in range(distance_matrix.shape[1]):
            order = distance_matrix[:, 0].argmin()
            sort_list.append(order)
            distance_matrix = distance_matrix[:, 1:]
            distance_matrix[order, :] = 2024
        return sort_list

    # ------------------------------
    # 5. Inference Loop
    # ------------------------------
    abs_results = []
    for x in loader:
        # For ImageNet, x might be (tensor, label)
        if args.dataset == 'imagenet':
            x, _ = x
        x = x.to(device)

        # ---------------------------------------------
        # (A) Optional puzzle step #1: local centercrop
        # If we trained with --crop, we do the 3×(64×64) splitting
        # This chunk splits the 192×192 image into 9 patches of (64×64),
        # optionally center-crops each patch to 64×64 again (no change),
        # then reassembles. Adjust if you do not want it.
        # ---------------------------------------------
        if args.dataset == 'imagenet' and args.crop:
            # We'll assume total image is 192×192 => 3×(64×64).
            centercrop = transforms.CenterCrop((64, 64))

            # Split into (3×64, 3×64) = 192, each patch is 64×64
            patchs = rearrange(
                x, 'b c (p1 h1) (p2 w1) -> b c (p1 p2) h1 w1',
                p1=3, p2=3, h1=64, w1=64
            )
            patchs = centercrop(patchs)  # effectively does nothing if each patch is 64 already
            x = rearrange(
                patchs, 'b c (p1 p2) h w -> b c (p1 h) (p2 w)',
                p1=3, p2=3, h=64, w=64
            )

        # ---------------------------------------------
        # (B) Shuffle puzzle step #2: random permutations
        # Now we do another rearrange with args.image_size//3
        # If image_size=192 => 192//3=64. That fits our puzzle logic.
        # ---------------------------------------------
        indices = np.random.permutation(9)

        # Break into 9 patches again, but we haven't changed the dims now
        # => x shape is [1, 3, 192, 192], so p1=3, p2=3, h1=64, w1=64
        x = rearrange(
            x, 'b c (p1 h1) (p2 w1) -> b c (p1 p2) h1 w1',
            p1=3, p2=3, h1=args.image_size//3, w1=args.image_size//3
        )
        # Shuffle patches
        x = x[:, :, indices, :, :]

        # Recombine into a scrambled puzzle
        x = rearrange(
            x, 'b c (p1 p2) h w -> b c (p1 h) (p2 w)',
            p1=3, p2=3, h=args.image_size//3, w=args.image_size//3
        )

        # ---------------------------------------------
        # (C) Diffusion-based puzzle solver
        # ---------------------------------------------
        samples = diffusion.p_sample_loop(
            model.forward,
            x,
            time_emb_noise.shape,
            time_emb_noise,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=device
        )

        # Evaluate each sample
        for sample_img, original_img in zip(samples, x):
            # The script reduces the final sample in some 2D manner:
            # each patch is further subdivided into ??? => "args.image_size // 48" => 192//48=4
            # That yields 4×4=16 sub-patches. The code is specialized for puzzle checks
            sample_reshaped = rearrange(
                sample_img,
                '(p1 h1 p2 w1) d -> (p1 p2) (h1 w1) d',
                p1=3, p2=3,
                h1=args.image_size // 48,
                w1=args.image_size // 48
            )

            # Average each patch
            sample_reshaped = sample_reshaped.mean(1)

            # Compare to time_emb
            dist = pairwise_distances(
                sample_reshaped.cpu().numpy(),
                time_emb[0].cpu().numpy(),
                metric='manhattan'
            )
            order = find_permutation(dist)
            pred = np.asarray(order).argsort()
            abs_results.append(int((pred == indices).all()))

        # Print stats each batch
        accuracy = np.asarray(abs_results).sum() / len(abs_results)
        print(f"test result on {len(abs_results)} samples is : {accuracy:.4f}")

        # Early stop
        if args.dataset == "met" and len(abs_results) >= 2000:
            break
        if args.dataset == "imagenet" and len(abs_results) >= 50000:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="JPDVT")
    parser.add_argument("--dataset", type=str, choices=["imagenet", "met"], default="imagenet")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--crop", action='store_true', default=False,
                        help="Whether to do puzzle splitting in code chunk (A). Matches how you trained with --crop.")
    parser.add_argument("--image-size", type=int, choices=[192, 288], default=192,
                        help="Must match your training resolution for no mismatch.")
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    # You must run with: --image-size=192 --crop  if that's how you trained.
    main(args)
