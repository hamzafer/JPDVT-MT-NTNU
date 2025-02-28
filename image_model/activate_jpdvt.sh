#!/bin/bash

# Load required modules
module purge
module load Anaconda3/2024.02-1
module load CUDA/12.4.0

# Activate your Conda environment
source activate /cluster/home/muhamhz/jpdvt_env

# Optional: print confirmation
echo "Environment activated successfully!"
