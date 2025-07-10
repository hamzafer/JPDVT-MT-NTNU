# With wandb (default)
torchrun --nnodes=1 --nproc_per_node=4 train_JPDVT.py --dataset imagenet --data-path /path/to/data

# Without wandb
torchrun --nnodes=1 --nproc_per_node=4 train_JPDVT.py --dataset imagenet --data-path /path/to/data --disable-wandb
