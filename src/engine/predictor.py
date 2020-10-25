import torch
import torchvision

from pathlib import Path
from PIL import Image
from omegaconf import DictConfig

from src.utils import build_model


class Predictor(object):
    def __init__(self, model_conf: DictConfig, weight_filepath: Path):
        self.model = build_model(model_conf=model_conf)

        if not weight_filepath.exists():
            raise RuntimeError(f"{str(weight_filepath)} not exist")

        self.model.load_state_dict(torch.load(str(weight_filepath)), strict=False)

    def __call__(self, x: torch.Tensor):
        return self.model.inference(x)

    def preprocess(self, image: Image):
        resize_image = torchvision.transforms.Resize((32, 32))(image)
        return torchvision.transforms.ToTensor()(resize_image).unsqueeze(0)

