import lightning as L
from typing import Any
from argparse import ArgumentParser
from omegaconf import OmegaConf

from models import MAE_Module, MAE, PrithviMAE
from data import PASTIS_S2_Module
from utils import get_data_sample, store_preds_as_images

_models = {
    "mae": MAE,
    "prithvi_mae": PrithviMAE,
}

_data_modules = {
    "pastis": PASTIS_S2_Module,
}


def generate_mae_predictions(config: dict[str, Any]) -> None:
    """
    Generate a prediction from a masked autoencoder (MAE).
    """
    if config.model.name not in _models:
        raise ValueError(
            f"Unsupported model type: {config.model.name}. Choose from: {list(_models.keys())}."
        )
    model_name = _models[config.model.name]
    model = MAE_Module(net=model_name, config=config)
    model = model_name.load_from_checkpoint(config.checkpoint.ckpt_path)
    
    # import torch
    # state_dict = torch.load(config.checkpoint.ckpt_path, map_location="cpu")
    
    # for key in list(state_dict.keys()):
    #     new_key = "net." + key
    #     state_dict[new_key] = state_dict.pop(key)
    
    # model.load_state_dict(state_dict, strict=False)
    
    if config.dataset.test.name not in _data_modules:
        raise ValueError(
            f"Unsupported dataset type: {config.dataset.test.name}. Choose from: {list(_data_modules.keys())}."
        )
    data_module = _data_modules[config.dataset.test.name](config)
    data_module.setup()
    subset_loader = get_data_sample(data_module.test_dataset, indices=[0, 5, 10, 15, 20])
    
    trainer = L.Trainer()
    trainer.test(model, dataloaders=subset_loader)
    predictions = trainer.predict(model, dataloaders=subset_loader)
    
    for batch in predictions:
        store_preds_as_images(batch, save_dir=config.checkpoint.save_dir)


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate predictions from a masked autoencoder (MAE).")
    parser.add_argument(
        "--config",
        help="Configurarion (.yaml) file to use",
        type=str,
        required=True
    )
    
    args = parser.parse_args()    
    config = OmegaConf.load(args.config)
    
    generate_mae_predictions(config)
    print("Predictions generated successfully.")