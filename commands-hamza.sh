# With wandb (default)
torchrun --nnodes=1 --nproc_per_node=4 train_JPDVT.py --dataset imagenet --data-path /path/to/data

# Without wandb
torchrun --nnodes=1 --nproc_per_node=4 train_JPDVT.py --dataset imagenet --data-path /path/to/data --disable-wandb


torchrun --nnodes=1 --nproc_per_node=4 train_JPDVT.py     --dataset texmet     --data-path /cluster/home/muhamhz/data/TEXMET     --image-size 288     --crop     --epochs 500     --ckpt-every 50000

