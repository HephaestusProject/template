"""
    This script was made by Nick at 19/07/20.
    To implement code for evaluating your model.
"""
import torch
import pytorch_lightning

pytorch_lightning.seed_everything(777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
