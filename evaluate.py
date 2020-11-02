"""
Usage:
    main.py evaluate [options] [--dataset-config=<dataset config path>] [--model-config=<model-config>] [--runner-config=<runner config path>] [--weight_filepath=<weight_filepath>]
    main.py evaluate (-h | --help)
Options:    
    --dataset-config <dataset config path>  Path to YAML file for dataset configuration  [default: conf/dataset/dataset.yml] [type: path]
    --model-config <model-config>  Path to YAML file for model configuration  [default: conf/model/model.yml] [type: path]    
    --runner-config <runner config path>  Path to YAML file for model configuration  [default: conf/runner/runner.yml] [type: path]
    --weight_filepath <weight_filepath>  Path to *.pth file for model weights  [default: pretrained_weights/LeNet_epoch=08-train_loss=0.05-val_loss=0.00-train_acc=0.98-val_acc=1.00.ckpt] [type: path]
    
    -h --help  Show this.
"""
import torch
from sklearn.metrics import classification_report

from src.engine.predictor import Predictor
from src.runner.runner import Runner
from src.utils import get_config, get_data_loaders


def evaluate(hparams: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_filepath = hparams.get("--weight_filepath")
    if not weight_filepath.exists():
        raise RuntimeError(f"{str(weight_filepath)} not exist")

    config_list: List = ["--dataset-config", "--model-config", "--runner-config"]
    config: DictConfig = get_config(hparams=hparams, options=config_list)

    _, test_dataloader = get_data_loaders(config=config)

    predictor = Predictor(model_conf=config.model)
    predictor.load_state_dict(torch.load(str(weight_filepath))["state_dict"], strict=True)
    predictor.eval()

    y_hat = []
    with torch.no_grad():
        for index, (data, _) in enumerate(test_dataloader):
            data = data.to(device)
            prediction = predictor.model(data)
            prediction = predictor.model.output_layer(prediction)
            labels = torch.argmax(prediction, dim=1).cpu().numpy()
            y_hat.extend(labels)

    report = classification_report(test_dataloader.dataset.targets, y_hat)
    print(report)
