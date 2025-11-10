import lightning as L
from typing import Any
from argparse import ArgumentParser
from omegaconf import OmegaConf
from pathlib import Path

from models import MAEModule, SegmentationModule
from data import PASTIS_S2_Module, IBGE_Module
from utils import get_data_sample, store_preds_as_images, load_from_checkpoint

_models = {
    "pretrain": MAEModule,
    "segmentation": SegmentationModule
}

_data_modules = {
    "pastis": PASTIS_S2_Module,
    "ibge": IBGE_Module
}


def test_and_predict(config: dict[str, Any]) -> None:
    """
    Test and generate predictions from a model.
    """
    if config.model.task not in _models:
        raise ValueError(
            f"Unsupported task type: {config.model.task}. Choose from: ['pretrain', 'segmentation']."
        )
    model_name = _models[config.model.task]
    model = model_name(config)
    model = load_from_checkpoint(config, model, model_name=model_name)
    
    # Setup data
    if config.dataset.name not in _data_modules:
        raise ValueError(
            f"Unsupported dataset type: {config.dataset.name}. Choose from: {list(_data_modules.keys())}."
        )
    data_module = _data_modules[config.dataset.name](config)
    data_module.setup()
    subset_loader = get_data_sample(data_module.test_dataset, indices=[0, 5, 10, 15, 20])
    
    trainer = L.Trainer()
    trainer.test(model, dataloaders=subset_loader)
    predictions = trainer.predict(model, dataloaders=subset_loader)
    
    for batch in predictions:
        store_preds_as_images(
            batch,
            save_dir=Path(config.checkpoint.save_dir) / config.checkpoint.run_name / f"version_{config.checkpoint.run_version}"
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Test and generate predictions from a segmentation model.")
    parser.add_argument(
        "--config",
        help="Configurarion (.yaml) file to use",
        type=str,
        required=True
    )
    
    args = parser.parse_args()    
    config = OmegaConf.load(args.config)
    
    test_and_predict(config)
    print("Predictions generated successfully.")