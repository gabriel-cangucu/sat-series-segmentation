import torch
import lightning as L
from argparse import ArgumentParser
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import OmegaConf
from typing import Any, Callable
from pathlib import Path

from models import MAEModule, SegmentationModule
from data import PASTIS_S2_Module, IBGE_Module, LEM_Module
from utils import load_from_checkpoint

_models = {
    "pretrain": MAEModule,
    "segmentation": SegmentationModule
}

_data_modules = {
    "pastis": PASTIS_S2_Module,
    "ibge": IBGE_Module,
    "lem": LEM_Module
}


def train_and_eval(config: dict[str, Any]) -> None:
    """
    Train a masked autoencoder (MAE) model for satellite series segmentation.
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

    logger = WandbLogger(
        save_dir=config.checkpoint.save_dir,
        name=config.checkpoint.run_name,
        id=None if not config.checkpoint.run_id else f"version_{config.checkpoint.run_id}",
        # default_hp_metric=False
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(config.checkpoint.save_dir) / "wandb" / f"ckpt_{config.checkpoint.run_name}",
        filename=f"{config.checkpoint.run_name}-{{epoch:02d}}-{{train_loss:.2f}}",
        monitor="train/loss" if config.model.task == "pretrain" else "val/loss",
        mode="min",
        save_top_k=2,
    )
    
    trainer = L.Trainer(
        max_epochs=config.solver.max_epochs,
        callbacks=[checkpoint_callback] if config.checkpoint.save_checkpoint else None,
        fast_dev_run=config.solver.dev_run,
        overfit_batches=1 if config.solver.overfit_batches else 0.,
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
    )
    
    ckpt_path = config.checkpoint.ckpt_path if config.checkpoint.ckpt_path else None
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = ArgumentParser(description="Train a masked autoencoder (MAE) model for satellite series segmentation.")
    parser.add_argument(
        "--config",
        help="Configurarion (.yaml) file to use",
        type=str,
        required=True
    )
    
    args = parser.parse_args()    
    config = OmegaConf.load(args.config)
    
    train_and_eval(config)
    print("Training completed successfully.")