import os
import io
import base64
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import sys
import torchvision
from torchvision import transforms
from einops import rearrange
from sklearn.metrics import pairwise_distances
import time

# Add the parent directory to PATH so we can import from image_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from existing model code
from image_model.models import DiT_models, get_2d_sincos_pos_embed  
from image_model.diffusion import create_diffusion

# Add imports at the top of your file
sys.path.append('/cluster/home/muhamhz/fcvit-mt-ntnu2')
from puzzle_fcvit import FCViT

###############################################################################
#                               CONFIGURATIONS
###############################################################################
# Model / Inference params (keeping the same as your original code)
MODEL_NAME = "JPDVT"
CHECKPOINT_PATH = "/cluster/home/muhamhz/JPDVT/image_model/models/3x3_Full/2850000.pt"
IMAGE_SIZE = 192
GRID_SIZE = 3
SEED = 0
NUM_SAMPLING_STEPS = 250

# System / Misc
BATCH_NORM_TRAIN_MODE = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Jigsaw Puzzle Solver API")

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

###############################################################################
#                               HELPER FUNCTIONS
###############################################################################
def center_crop_arr(pil_image, image_size):
    """Center cropping implementation from your existing code."""
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
    """Greedy algorithm to find the permutation order based on the distance matrix."""
    sort_list = []
    tmp = np.copy(distance_matrix)
    for _ in range(tmp.shape[1]):
        order = tmp[:, 0].argmin()
        sort_list.append(order)
        tmp = tmp[:, 1:]
        # effectively remove that row so it's not chosen again
        tmp[order, :] = 1e9  
    return sort_list

def tensor_to_base64(tensor, nrow=1, normalize=True):
    """Convert a tensor to a base64 encoded image"""
    if normalize:
        tensor = tensor * 0.5 + 0.5  # Unnormalize
        tensor = tensor.clamp(0, 1)
    
    if tensor.dim() == 4 and nrow > 1:  # multiple images
        grid = torchvision.utils.make_grid(tensor, nrow=nrow, normalize=False)
        pil_image = transforms.ToPILImage()(grid)
    else:  # single image
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        pil_image = transforms.ToPILImage()(tensor)
    
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

###############################################################################
#                       MODEL LOADING (AT STARTUP)
###############################################################################
# Global variables for the model and diffusion
model = None
diffusion = None
time_emb = None
time_emb_noise = None
transform = None

@app.on_event("startup")
async def load_model():
    """Load the model when the API starts up."""
    global model, diffusion, time_emb, time_emb_noise, transform
    
    print(f"Loading model [{MODEL_NAME}] from checkpoint: {CHECKPOINT_PATH}")
    
    # Set random seed
    torch.manual_seed(SEED)
    torch.set_grad_enabled(False)
    
    # Initialize transform
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    
    # Load model
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
    
    print("Model and diffusion initialized successfully!")

###############################################################################
#                          API ENDPOINTS
###############################################################################
class SolveRequest(BaseModel):
    image_data: str
    model_id: str = "default"
    indices: list[int] = None
    
    model_config = {
        "protected_namespaces": ()
    }

@app.get("/")
async def root():
    """Redirect to index.html"""
    return RedirectResponse(url="/index.html")

@app.get("/api/models")
async def get_models():
    """Get available models."""
    return [
        {
            "id": "default",
            "name": "JPDVT",
            "description": "3x3 Grid Jigsaw Puzzle Solver"
        },
        {
            "id": "fcvit",
            "name": "FCViT",
            "description": "Fragment Contextual Vision Transformer Puzzle Solver"
        }
    ]

@app.post("/api/create_puzzle")
async def create_puzzle(file: UploadFile = File(...), seed: int = Form(None)):
    """
    Create a scrambled puzzle from an uploaded image.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    try:
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Read and process image
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        x = transform(pil_img).unsqueeze(0).to(DEVICE)
        
        # Original image for reference
        original_base64 = tensor_to_base64(x)
        
        # Scramble the puzzle - precisely following the notebook approach
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
        
        # Convert scrambled image to base64
        puzzle_base64 = tensor_to_base64(x_scrambled)
        
        # Calculate initial metrics based on scrambling
        # For a completely scrambled puzzle, patch accuracy can be calculated
        patch_matches = (indices == np.arange(GRID_SIZE * GRID_SIZE)).sum()
        total_patches = GRID_SIZE * GRID_SIZE
        patch_accuracy = patch_matches / total_patches
        
        return {
            "original_image": original_base64,
            "puzzle_image": puzzle_base64,
            "indices": indices.tolist(),
            "initial_metrics": {
                "patch_matches": int(patch_matches),
                "total_patches": total_patches,
                "patch_accuracy": float(patch_accuracy),
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating puzzle: {str(e)}")

@app.post("/api/solve_puzzle")
async def solve_puzzle(file: UploadFile = File(...)):
    """
    Solve a jigsaw puzzle from an uploaded image.
    This endpoint directly processes an image file.
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded yet")
    
    try:
        # Read and preprocess image
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        x = transform(pil_img).unsqueeze(0).to(DEVICE)
        
        # Original image (for reference)
        original_unnorm = x.clone()
        
        # Scramble the puzzle
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
        
        # Run diffusion model
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
        
        # Predict permutation
        sample_patch_dim = IMAGE_SIZE // (16 * GRID_SIZE)
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
        
        # Metrics
        puzzle_correct = int((pred == indices).all())
        patch_matches = (pred == indices).sum()
        total_patches = GRID_SIZE * GRID_SIZE
        patch_accuracy = patch_matches / total_patches
        
        # Reconstruct final puzzle
        scrambled_patches_list = [x_patches[0, :, i, :, :] for i in range(total_patches)]
        reconstructed_patches = [None] * total_patches
        for i, pos in enumerate(pred):
            reconstructed_patches[pos] = scrambled_patches_list[i]
        grid_reconstructed = torch.stack(reconstructed_patches)
        
        # Convert to base64 for response
        original_base64 = tensor_to_base64(original_unnorm)
        scrambled_base64 = tensor_to_base64(x_scrambled)
        reconstructed_base64 = tensor_to_base64(grid_reconstructed, nrow=GRID_SIZE)
        
        return {
            "success": True,
            "original_image": original_base64,
            "scrambled_image": scrambled_base64,
            "solution_image": reconstructed_base64,
            "metrics": {
                "puzzle_correct": puzzle_correct,
                "patch_matches": int(patch_matches),
                "total_patches": total_patches,
                "patch_accuracy": float(patch_accuracy),
            },
            "details": {
                "indices": indices.tolist(),
                "predicted_order": pred.tolist(),
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error solving puzzle: {str(e)}")

@app.post("/api/solve")
async def solve(data: SolveRequest):
    """
    Solve a jigsaw puzzle from a base64-encoded image.
    """
    # Check if model_id is fcvit
    if data.model_id == "fcvit":
        return await solve_with_fcvit(data.image_data, data.indices)
    
    # Default JPDVT model logic
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded yet")
    
    try:
        # Track execution time
        start_time = time.time()
        
        # Decode base64 image
        image_data = base64.b64decode(data.image_data)
        pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")
        x_scrambled = transform(pil_img).unsqueeze(0).to(DEVICE)
        
        # Get original indices from the request - these are the scrambling indices
        original_indices = np.array(data.indices)
        
        # Run diffusion model
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
        
        # Extract patches from the scrambled image
        x_patches = rearrange(
            x_scrambled, 'b c (p1 h1) (p2 w1) -> b c (p1 p2) h1 w1',
            p1=GRID_SIZE, p2=GRID_SIZE,
            h1=IMAGE_SIZE//GRID_SIZE, w1=IMAGE_SIZE//GRID_SIZE
        )
        
        # Extract feature representations
        sample_patch_dim = IMAGE_SIZE // (16 * GRID_SIZE)
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
        
        # Calculate metrics directly comparing predicted order with original indices
        # This is exactly how the notebook does it
        puzzle_correct = int((pred == original_indices).all())
        patch_matches = (pred == original_indices).sum()
        total_patches = GRID_SIZE * GRID_SIZE
        patch_accuracy = patch_matches / total_patches
        
        # Reconstruct final puzzle
        scrambled_patches_list = [x_patches[0, :, i, :, :] for i in range(total_patches)]
        reconstructed_patches = [None] * total_patches
        for i, pos in enumerate(pred):
            reconstructed_patches[pos] = scrambled_patches_list[i]
        grid_reconstructed = torch.stack(reconstructed_patches)
        
        # Create reconstructed image
        reconstructed_base64 = tensor_to_base64(grid_reconstructed, nrow=GRID_SIZE)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        return {
            "success": True,
            "solution_image": reconstructed_base64,
            "predicted_order": pred.tolist(),
            "metrics": {
                "puzzle_correct": puzzle_correct,
                "patch_matches": int(patch_matches), 
                "total_patches": total_patches,
                "patch_accuracy": float(patch_accuracy),  
            },
            "image_info": {
                "grid_size": f"{GRID_SIZE}x{GRID_SIZE}",
                "image_resolution": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
                "patch_size": f"{IMAGE_SIZE//GRID_SIZE}x{IMAGE_SIZE//GRID_SIZE}"
            },
            "processing_time": round(elapsed_time, 2)
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error solving puzzle: {str(e)}")

# Add this function to handle FCViT inference
async def solve_with_fcvit(image_data, indices=None):
    # Constants from infer_visualise.py
    CKPT_PATH = "/cluster/home/muhamhz/fcvit-mt-ntnu/checkpoint/FCViT_base_3x3_ep100_lr3e-05_b64.pt"
    BACKBONE = "vit_base_patch16_224"
    PUZZLE_SIZE = 225
    FRAG_SIZE = 75
    NUM_FRAG = 9
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Track execution time
    start_time = time.time()
    
    # Prepare model
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    state = {k.replace("module.", "", 1): v for k, v in ckpt["model"].items()}
    
    model = FCViT(backbone=BACKBONE, num_fragment=NUM_FRAG, size_fragment=FRAG_SIZE).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    model.augment_fragment = transforms.Resize((FRAG_SIZE, FRAG_SIZE), antialias=True)
    
    # Process image
    try:
        img_data = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        
        tfs = transforms.Compose([
            transforms.Resize((PUZZLE_SIZE, PUZZLE_SIZE), antialias=True),
            transforms.ToTensor()
        ])
        img_tensor = tfs(img).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            pred_gpu, tgt_gpu = model(img_tensor)
        
        # Move prediction to CPU before comparison
        pred_ = model.mapping(pred_gpu.clone()).cpu()
        map_coord = model.map_coord.cpu()
        
        # Get predicted order - now both tensors are on CPU
        mask_pred = (pred_[0][:, None, :] == map_coord).all(-1).long()
        pred_order = mask_pred.argmax(dim=1).tolist()
        
        # Create reconstructed image
        def unshuffle(tensor, order):
            C, H, W = tensor.shape
            p = FRAG_SIZE
            pieces = [tensor[:, i:i+p, j:j+p] for i in range(0, H, p) for j in range(0, W, p)]
            grid = [pieces[idx] for idx in order]
            rows = [torch.cat(grid[i:i+3], dim=2) for i in range(0, 9, 3)]
            return torch.cat(rows, dim=1)
        
        # Move image tensor to CPU for reconstruction
        img_cpu = img_tensor[0].cpu()
        reconstructed = unshuffle(img_cpu, pred_order)
        
        # Convert to image
        result_img = transforms.ToPILImage()(reconstructed)
        buffered = io.BytesIO()
        result_img.save(buffered, format="PNG")
        solution_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Calculate metrics
        if indices and isinstance(indices, list) and len(indices) == NUM_FRAG:
            patch_matches = sum(1 for i, j in zip(pred_order, indices) if i == j)
            patch_accuracy = patch_matches / NUM_FRAG
            puzzle_correct = patch_matches == NUM_FRAG
        else:
            patch_matches = 0
            patch_accuracy = 0
            puzzle_correct = False
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Return similar format to your existing API
        return {
            "success": True,
            "solution_image": solution_image,
            "predicted_order": pred_order,  # Added this to match JPDVT output format
            "metrics": {
                "puzzle_correct": puzzle_correct,
                "patch_matches": int(patch_matches),
                "total_patches": NUM_FRAG,
                "patch_accuracy": float(patch_accuracy)
            },
            "image_info": {
                "grid_size": "3x3",
                "image_resolution": f"{PUZZLE_SIZE}x{PUZZLE_SIZE}",
                "patch_size": f"{FRAG_SIZE}x{FRAG_SIZE}"
            },
            "processing_time": round(elapsed_time, 2)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error solving puzzle with FCViT: {str(e)}")

# Serve static files for frontend
app.mount("/", StaticFiles(directory="api/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
