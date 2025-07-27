# With wandb (default)
torchrun --nnodes=1 --nproc_per_node=4 train_JPDVT.py --dataset imagenet --data-path /path/to/data

# Without wandb
torchrun --nnodes=1 --nproc_per_node=4 train_JPDVT.py --dataset imagenet --data-path /path/to/data --disable-wandb


torchrun --nnodes=1 --nproc_per_node=4 train_JPDVT.py     --dataset texmet     --data-path /cluster/home/muhamhz/data/TEXMET     --image-size 288     --crop     --epochs 500     --ckpt-every 50000

# Multi-GPU finetuning (adjust nproc_per_node to your GPU count)
torchrun --nproc_per_node=4 train_JPDVT.py \
    --data-path /cluster/home/muhamhz/data/TEXMET \
    --dataset texmet \
    --image-size 192 \
    --epochs 500 \
    --ckpt /cluster/home/muhamhz/JPDVT/image_model/models/3x3_Full/2850000.pt \
    --results-dir results_finetune \
    --log-every 50 \
    --ckpt-every 2500 \
    --wandb-tag "FINETUNE_TEXMET"
