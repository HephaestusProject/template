"""
    This script was made by Nick at 19/07/20.
    To implement code for inference with your model.
"""
import pytorch_lightning
import torch

pytorch_lightning.seed_everything(777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
