import torch
import lightning as L
from argparse import ArgumentParser
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from omegaconf import OmegaConf
from typing import Any

from models import PrithviMAE
from data import PASTIS_S2_Module, IBGE_Module

_models = {
    "prithvi_mae": PrithviMAE,
}

_data_modules = {
    "pastis": PASTIS_S2_Module,
    "ibge": IBGE_Module
}


def pretrain(config: dict[str, Any]) -> None:
    """
    Pretrain a masked autoencoder (MAE) model for satellite series.
    """
    if config.model.name not in _models:
        raise ValueError(
            f"Unsupported model type: {config.model.name}. Choose from: {list(_models.keys())}."
        )
    model = _models[config.model.name](config)
    
    if config.dataset.train.name not in _data_modules:
        raise ValueError(
            f"Unsupported dataset type: {config.dataset.train.name}. Choose from: {list(_data_modules.keys())}."
        )
    data_module = _data_modules[config.dataset.train.name](config)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint.save_dir,
        filename=f"{config.checkpoint.run_name}-{{epoch:02d}}-{{train_loss:.2f}}",
        monitor="train_loss",
        mode="min",
        save_top_k=1,
    ) if config.checkpoint.save_checkpoint else None
    
    logger = TensorBoardLogger(
        save_dir=config.checkpoint.save_dir,
        name=config.checkpoint.run_name,
        default_hp_metric=False
    )
    
    trainer = L.Trainer(
        max_epochs=config.solver.max_epochs,
        callbacks=[checkpoint_callback] if config.checkpoint.save_checkpoint else None,
        fast_dev_run=config.solver.dev_run,
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto"
    )
    
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    parser = ArgumentParser(description="Pretrain a masked autoencoder (MAE) model for satellite series.")
    parser.add_argument(
        "--config",
        help="Configurarion (.yaml) file to use",
        type=str,
        required=True
    )
    
    args = parser.parse_args()    
    config = OmegaConf.load(args.config)
    
    pretrain(config)
    print("Pretraining completed successfully.")