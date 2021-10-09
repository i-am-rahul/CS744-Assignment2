import torch.distributed as dist
import os
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
import torch
import model as mdl
import torch.distributed as dist

NUM_THREADS = 5
backend = "gloo"
master_ip = "10.10.1.1"
master_port = 29500
rank = 0
world_size = 1
batch_size = int(256/world_size)
device="cpu"
torch.set_num_threads(NUM_THREADS)
print_every_iteration = 20

def main():


if __name__ == "__main__":
    main()
