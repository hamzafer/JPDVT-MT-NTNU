"""
Solve Jigsaw Puzzles with JPDVT
"""
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

def imshow_tensor(img_tensor, title=""):
    # Unnormalize the image (assuming it was normalized with mean=0.5, std=0.5)
    img_tensor = img_tensor * 0.5 + 0.5
    npimg = img_tensor.permute(1, 2, 0).cpu().numpy()
    plt.imshow(np.clip(npimg, 0, 1))
    plt.title(title)
    plt.axis("off")
    plt.show()

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

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda"
    
    # (Template code remains unchanged)
    template = np.zeros((6,6))
    for i in range(6):
        for j in range(6):
            template[i,j] = 18 * i + j
    template = np.concatenate((template,template,template),axis=0)
    template = np.concatenate((template,template,template),axis=1)

    # Load model:
    model = DiT_models[args.model](
        input_size=args.image_size,
    ).to(device)
    print("Load model from:", args.ckpt )
    ckpt_path = args.ckpt 
    model_dict = model.state_dict()
    state_dict = torch.load(ckpt_path, weights_only=False)
    model_state_dict = state_dict['model']
    pretrained_dict = {k: v for k, v in model_state_dict.items() if k in model_dict}
    model.load_state_dict(pretrained_dict, strict=False)

    print("Model keys:", list(model_dict.keys())[:10])
    print("Checkpoint keys:", list(model_state_dict.keys()))

    transform = transforms.Compose([
       transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 192)),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    # Because the batchnorm doesn't work normally when batch size is 1
    # Thus we set the model to train mode
    model.train() 

    diffusion = create_diffusion(str(args.num_sampling_steps))
    if args.dataset == "met":
        dataset = MET(args.data_path, 'test')
    elif args.dataset == "imagenet":
        dataset = ImageFolder(args.data_path, transform=transform)
    
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )

    time_emb = torch.tensor(get_2d_sincos_pos_embed(8, args.grid_size)).unsqueeze(0).float().to(device)
    time_emb_noise = torch.tensor(get_2d_sincos_pos_embed(8, 12)).unsqueeze(0).float().to(device)
    time_emb_noise = torch.randn_like(time_emb_noise)
    time_emb_noise = time_emb_noise.repeat(1,1,1)
    model_kwargs = None

    # Greedy algorithm to find permutation order
    def find_permutation(distance_matrix):
        sort_list = []
        for m in range(distance_matrix.shape[1]):
            order = distance_matrix[:,0].argmin()
            sort_list.append(order)
            distance_matrix = distance_matrix[:,1:]
            distance_matrix[order,:] = 2024
        return sort_list

    abs_results = []
    # For a variable grid, let G be the grid size.
    G = args.grid_size

    for x in loader:
        if args.dataset == 'imagenet':
            x, _ = x
        x = x.to(device)
        imshow_tensor(x[0], title="Original Image")
        
        if args.dataset == 'imagenet' and args.crop:
            centercrop = transforms.CenterCrop((64,64))
            patchs = rearrange(x, 'b c (p1 h1) (p2 w1) -> b c (p1 p2) h1 w1', 
                               p1=G, p2=G, h1=args.image_size//G, w1=args.image_size//G)
            patchs = centercrop(patchs)
            x = rearrange(patchs, 'b c (p1 p2) h1 w1 -> b c (p1 h1) (p2 w1)', 
                          p1=G, p2=G, h1=args.image_size//G, w1=args.image_size//G)
        
        # Generate the puzzles: split the image into GxG patches
        indices = np.random.permutation(G * G)
        print("Random permutation indices:", indices)
        x = rearrange(x, 'b c (p1 h1) (p2 w1) -> b c (p1 p2) h1 w1', 
                      p1=G, p2=G, h1=args.image_size//G, w1=args.image_size//G)
        
        # Save the patches before any permutation
        patches_before = [x[0, :, i, :, :] for i in range(G * G)]
        grid = torch.stack(patches_before)
        save_image(grid, "debug_patches_before.png", nrow=G, normalize=True)
        plt.figure(figsize=(4, 4))
        plt.imshow(torchvision.utils.make_grid(grid, nrow=G, normalize=True).permute(1, 2, 0).cpu().numpy())
        plt.title("Patches Before Permutation")
        plt.axis("off")
        plt.show()
        
        # Permute the patches
        x = x[:, :, indices, :, :]
        # Save scrambled patches for later visualization of the final puzzle
        scrambled_patches = [x[0, :, i, :, :] for i in range(G * G)]
        grid = torch.stack(scrambled_patches)
        save_image(grid, "debug_patches_after.png", nrow=G, normalize=True)
        plt.figure(figsize=(4, 4))
        plt.imshow(torchvision.utils.make_grid(grid, nrow=G, normalize=True).permute(1, 2, 0).cpu().numpy())
        plt.title("Patches After Permutation")
        plt.axis("off")
        plt.show()
        
        # Reassemble the scrambled patches into the final scrambled image
        x = rearrange(x, 'b c (p1 p2) h1 w1 -> b c (p1 h1) (p2 w1)', 
                      p1=G, p2=G, h1=args.image_size//G, w1=args.image_size//G)
        imshow_tensor(x[0], title="Final Scrambled Image")
        print("Scrambled image shape:", x.shape)
        
        # Generate samples using the diffusion process
        samples = diffusion.p_sample_loop(
            model.forward, x, time_emb_noise.shape, time_emb_noise, 
            clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        print("Samples shape:", samples.shape)
        
        for sample, img in zip(samples, x):
            print("Raw sample shape:", sample.shape)
            # For the sample reordering, we use a downsampled patch size.
            sample_patch_dim = args.image_size // (16 * G)  # For grid=3, 192//48=4
            sample = rearrange(sample, '(p1 h1 p2 w1) d -> (p1 p2) (h1 w1) d', 
                                 p1=G, p2=G, h1=sample_patch_dim, w1=sample_patch_dim)
            print("Rearranged sample shape:", sample.shape)
            sample = sample.mean(1)
            dist = pairwise_distances(sample.cpu().numpy(), time_emb[0].cpu().numpy(), metric='manhattan')
            order = find_permutation(dist)
            pred = np.asarray(order).argsort()
            print("Predicted permutation:", pred)
            abs_results.append(int((pred == indices).all()))
            
            # --- Final Puzzle Visualization ---
            # Reconstruct the final puzzle by reordering the scrambled patches using the predicted order.
            reconstructed_patches = [None] * (G * G)
            for i, pos in enumerate(pred):
                reconstructed_patches[pos] = scrambled_patches[i]
            grid_reconstructed = torch.stack(reconstructed_patches)
            save_image(grid_reconstructed, "grid_reconstructed.png", nrow=G, normalize=True)
            plt.figure(figsize=(4, 4))
            plt.imshow(torchvision.utils.make_grid(grid_reconstructed, nrow=G, normalize=True).permute(1, 2, 0).cpu().numpy())
            plt.title("Final Reconstructed Puzzle")
            plt.axis("off")
            plt.show()
            # --- End of Final Puzzle Visualization ---
         
        print("Test result on", len(abs_results), "samples is:", np.asarray(abs_results).sum()/len(abs_results))
        break
        if len(abs_results) >= 2000 and args.dataset == "met":
            break
        if len(abs_results) >= 50000 and args.dataset == "imagenet":
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="JPDVT")
    parser.add_argument("--dataset", type=str, choices=["imagenet", "met"], default="imagenet")
    parser.add_argument("--data-path", type=str, default="train", help="Subpath inside /cluster/home/muhamhz/data/imagenet/")
    parser.add_argument("--crop", action='store_true', default=False)
    parser.add_argument("--image-size", type=int, choices=[192, 288], default=192)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default="/cluster/home/muhamhz/JPDVT/image_model/results/009-imagenet-JPDVT-crop/checkpoints/2850000.pt", help="Default checkpoint path")
    # New argument to set grid size (e.g., 3 for 3x3, 4 for 4x4, 5 for 5x5)
    parser.add_argument("--grid-size", type=int, default=3, help="Grid size for the jigsaw puzzle (number of patches per row/column)")
    
    args = parser.parse_args()

    # Construct full data path
    base_data_path = "/cluster/home/muhamhz/data/imagenet/"
    full_data_path = os.path.join(base_data_path, args.data_path)

    print(f"Using data path: {full_data_path}")
    print(f"Using checkpoint: {args.ckpt}")

    # Pass full data path instead of args.data_path
    args.data_path = full_data_path

    main(args)
