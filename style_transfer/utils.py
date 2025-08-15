import torch
import random
import numpy as np

def calc_mean_std(feat: torch.Tensor, eps: float = 1e-5):
    """Calculate channel-wise mean and standard deviation"""
    N, C = feat.size()[:2] # N = batch size, C = channels
    feat_var = feat.view(N, C, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
