import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
import torchvision.transforms as T
from einops import rearrange
from einops.layers.torch import Rearrange

from torchvision import transforms
from sklearn.model_selection import train_test_split
import cv2

# Fix PIL DecompressionBombError for large images
Image.MAX_IMAGE_PIXELS = None  # Remove the limit
# Or set a higher limit if you prefer: Image.MAX_IMAGE_PIXELS = 500000000

class MET(Dataset):
    def __init__(self, image_dir,split):
        seed = 42
        torch.manual_seed(seed)
        self.split = split

        all_files = os.listdir(image_dir)
        self.image_files = [os.path.join(image_dir,all_files[0])+'/' + k for k in os.listdir(os.path.join(image_dir,all_files[0]))]
        self.image_files += [os.path.join(image_dir,all_files[1])+'/' + k for k in os.listdir(os.path.join(image_dir,all_files[1]))]
        self.image_files += [os.path.join(image_dir,all_files[2])+'/' + k for k in os.listdir(os.path.join(image_dir,all_files[2]))]
        # +os.listdir(os.path.join(image_dir,all_files[1]))+os.listdir(os.path.join(image_dir,all_files[2]))
        for image in self.image_files:
            if '.jpg' not in image:
                self.image_files.remove(image)
        dataset_indices = list(range(len( self.image_files)))

        train_indices, test_indices = train_test_split(dataset_indices, test_size=2000, random_state=seed)
        train_indices, val_indices = train_test_split(train_indices, test_size=1000, random_state=seed) 
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.val_indices = val_indices

        # Define the color jitter parameters
        brightness = 0.4  # Randomly adjust brightness with a maximum factor of 0.4
        contrast = 0.4    # Randomly adjust contrast with a maximum factor of 0.4
        saturation = 0.4  # Randomly adjust saturation with a maximum factor of 0.4
        hue = 0.1         # Randomly adjust hue with a maximum factor of 0.1

        # Because the Met is a much more smaller dataset, so we use more complex data augmentation here
        flip_probability = 0.5
        self.transform1 = transforms.Compose([
            transforms.Resize(398), 
            transforms.RandomCrop((398,398)),
            transforms.RandomHorizontalFlip(p=flip_probability),  # Horizontal flipping with 0.5 probability
            transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

        self.transform2 = transforms.Compose([
            transforms.Resize(398), 
            transforms.CenterCrop((398,398)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

    def __len__(self):
        if self.split == 'train':
            return len(self.train_indices)
        elif self.split == 'val':
            return len(self.val_indices)
        elif self.split == 'test':
            return len(self.test_indices)

    def rand_erode(self,image,n_patches):
        output = torch.zeros(3,96*3,96*3)
        crop = transforms.RandomCrop((96,96))
        gap = 48
        patch_size = 100
        for i in range(n_patches):
            for j in range(n_patches):
                left = i * (patch_size + gap)
                upper = j * (patch_size + gap)
                right = left + patch_size
                lower = upper + patch_size

                patch = crop(image[:,left:right, upper:lower])
                output[:,i*96:i*96+96,j*96:j*96+96] = patch

        return output    

    def __getitem__(self, idx):
        if self.split == 'train':
            index = self.train_indices[idx]
            image = self.transform1(Image.open(self.image_files[index]))
            image = self.rand_erode(image,3)
        elif self.split == 'val':
            index = self.val_indices[idx]
            image = self.transform2(Image.open(self.image_files[index]))
            image = self.rand_erode(image,3)
        elif self.split == 'test':
            index = self.test_indices[idx]
            image = self.transform2(Image.open(self.image_files[index]))
            image = self.rand_erode(image,3)

        return image

class TEXMET(Dataset):
    def __init__(self, data_dir, split):
        self.split = split
        self.data_dir = data_dir
        
        # Load image files from the corresponding split text file
        split_file = os.path.join(data_dir, f"{split}_files.txt")
        
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            # Read all lines and strip whitespace
            image_filenames = [line.strip() for line in f.readlines()]
        
        # Convert filenames to absolute paths
        # All images are directly in /cluster/home/muhamhz/data/TEXMET/images/
        images_dir = os.path.join(data_dir, "images")
        
        # Extract just the filename from paths like "train/images/file.jpg"
        self.image_files = []
        for img_path in image_filenames:
            # Get just the filename (last part after /)
            filename = os.path.basename(img_path)
            # Create full path to the actual image location
            full_path = os.path.join(images_dir, filename)
            self.image_files.append(full_path)
        
        # Filter out non-existent files
        existing_files = []
        missing_count = 0
        for img_path in self.image_files:
            if os.path.exists(img_path):
                existing_files.append(img_path)
            else:
                missing_count += 1
                if missing_count <= 5:  # Only print first 5 missing files
                    print(f"Warning: File not found: {img_path}")
        
        if missing_count > 5:
            print(f"Warning: {missing_count - 5} more files not found...")
        
        self.image_files = existing_files
        print(f"TEXMET {split} split: {len(self.image_files)} images loaded successfully")
        if missing_count > 0:
            print(f"TEXMET {split} split: {missing_count} images missing")

        # Define the color jitter parameters (adjusted for textile images)
        brightness = 0.3  # Slightly less aggressive for textile textures
        contrast = 0.3    
        saturation = 0.3  
        hue = 0.05        # Very small hue changes for textiles

        # Safe image loading with initial resize for very large images
        def safe_resize(img):
            """Safely resize very large images before other transforms"""
            # If image is extremely large, first resize to manageable size
            max_size = 2048
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.LANCZOS)
            return img

        # Data augmentation for training
        flip_probability = 0.5
        self.transform1 = transforms.Compose([
            transforms.Lambda(safe_resize),  # Safe resize first
            transforms.Resize(398), 
            transforms.RandomCrop((398,398)),
            transforms.RandomHorizontalFlip(p=flip_probability),
            transforms.RandomVerticalFlip(p=0.2),  # Add vertical flip for textiles
            transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

        # No augmentation for validation/test
        self.transform2 = transforms.Compose([
            transforms.Lambda(safe_resize),  # Safe resize first
            transforms.Resize(398), 
            transforms.CenterCrop((398,398)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

    def __len__(self):
        return len(self.image_files)

    def rand_erode(self, image, n_patches):
        """Create puzzle-like patches with gaps between them"""
        output = torch.zeros(3, 96*3, 96*3)
        crop = transforms.RandomCrop((96, 96))
        gap = 48
        patch_size = 100
        
        for i in range(n_patches):
            for j in range(n_patches):
                left = i * (patch_size + gap)
                upper = j * (patch_size + gap)
                right = left + patch_size
                lower = upper + patch_size

                patch = crop(image[:, left:right, upper:lower])
                output[:, i*96:i*96+96, j*96:j*96+96] = patch

        return output    

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        try:
            # Open image with proper error handling
            with Image.open(img_path) as img:
                # Convert to RGB to handle different formats
                img = img.convert('RGB')
                
                # Apply appropriate transform based on split
                if self.split == 'train':
                    image = self.transform1(img)
                else:
                    image = self.transform2(img)
                
                # Apply rand_erode to create puzzle patches
                image = self.rand_erode(image, 3)
                
                return image
                
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a black image as fallback
            return torch.zeros(3, 288, 288)
