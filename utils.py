import random
import numpy as np
import torch
import os

def setup_seed(seed=42, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[INFO] Seed set to: {seed}")

def get_device(args):
    return torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")