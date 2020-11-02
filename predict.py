"""
Usage:
    main.py predict [options] [--model_config=<model_config>] [--weight_filepath=<weight_filepath>] [--image_filepath=<image_filepath>]
    main.py predict (-h | --help)
Options:    
    --model_config <model_config>  Path to YAML file for model configuration  [default: conf/model/model.yml] [type: path]
    --weight_filepath <weight_filepath>  Path to *.pth file for model weights  [default: pretrained_weights/LeNet_epoch=01-train_loss=0.10-val_loss=0.02-train_acc=0.97-val_acc=1.00.ckpt] [type: path]
    --image_filepath <image_filepath>  Path to image file for inference  [default: pretrained_weights/sample.png] [type: path]
    
    
    -h --help  Show this.
"""
from pathlib import Path
from typing import Dict

import torch
from PIL import Image

from src.engine.predictor import Predictor
from src.utils import get_config


def predict(hparams: Dict):
    config_list: List = ["--model_config"]

    config: DictConfig = get_config(hparams=hparams, options=config_list)

    weight_filepath = hparams.get("--weight_filepath")
    image_filepath = hparams.get("--image_filepath")

    predictor = Predictor(model_conf=config.model)
    predictor.load_state_dict(
        torch.load(str(weight_filepath))["state_dict"], strict=True
    )
    predictor.eval()

    image = Image.open(str(image_filepath))
    image: torch.Tensor = predictor.preprocess(image=image)

    prediction = predictor(image)

    print(f"Result of inference : `{prediction}`")
