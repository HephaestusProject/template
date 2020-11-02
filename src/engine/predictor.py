from pathlib import Path

import torch
import torchvision
from omegaconf import DictConfig
from PIL import Image

from src.utils import build_model


class Predictor(torch.nn.Module):
    def __init__(self, model_conf: DictConfig):
        super().__init__()
        self.model = build_model(model_conf=model_conf)

    def __call__(self, x: torch.Tensor):
        return self.model.inference(x)

    def preprocess(self, image: Image):
        resize_image = torchvision.transforms.Resize((32, 32))(image)
        return torchvision.transforms.ToTensor()(resize_image).unsqueeze(0)
