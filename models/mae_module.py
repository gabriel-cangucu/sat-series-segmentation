import torch
import lightning as L
from torch import nn
from typing import Any
from lightly.utils.debug import std_of_l2_normalized

from utils import get_random_embedding
from .backbones.mae import MAE
from .backbones.prithvi_mae import PrithviMAE
from .backbones.swin_mae import SwinMAE

_backbones = {
    "mae": MAE,
    "prithvi_mae": PrithviMAE,
    "swin_mae": SwinMAE
}

_solvers = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW
}


class MAEModule(L.LightningModule):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        
        if config.model.backbone not in _backbones:
            raise ValueError(
                f"Unsupported model type: {config.model.backbone}. Choose from: {list(_backbones.keys())}."
            )
        
        self.config = config
        self.save_hyperparameters()
        
        backbone = _backbones[config.model.backbone]
        self.backbone = backbone(
            img_size=config.model.img_size,
            patch_size=config.model.patch_size,
            in_chans=config.model.num_channels,
            num_frames=config.model.num_frames
        )

    def forward(self,
        x: torch.Tensor,
        temporal_coords: None | torch.Tensor = None,
        mask_ratio: None | float = 0.75
    ) -> torch.Tensor:
        return self.backbone(x, mask_ratio=mask_ratio)
    
    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        outputs = self(batch["data"], temporal_coords=batch["dates"])
        
        latent = get_random_embedding(outputs["latents"])
        embeddings_std = torch.nan_to_num(std_of_l2_normalized(latent))
        learning_rate = self.optimizers().param_groups[0]["lr"]
        
        self.log("train/loss", outputs["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("embeddings_std", embeddings_std, on_step=False, on_epoch=True, prog_bar=True)
        self.log("lr", learning_rate, on_step=False, on_epoch=True, prog_bar=True)
        
        return outputs["loss"]
    
    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self(batch["data"], temporal_coords=batch["dates"])
        
        self.log("test/loss", outputs["loss"], on_step=False, on_epoch=True, prog_bar=True)
    
    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self(batch["data"], temporal_coords=batch["dates"])
    
    def configure_optimizers(self) -> dict[str, Any]:
        def warmup_cosine_lr_lambda(curr_epoch: int, warmup_epochs: int = 10, max_epochs: int = 100):
            '''
            Custom function for cosine scheduler with linear warmup epochs
            '''
            if curr_epoch < warmup_epochs:
                return (curr_epoch + 1) / max(1, warmup_epochs)  # Linear warmup

            progress = torch.tensor(curr_epoch - warmup_epochs) / (max_epochs - warmup_epochs)
            return 0.5 * (1 + torch.cos(torch.pi * progress)).item()

        if self.config.solver.name not in _solvers:
            raise ValueError(
                f"Solver {self.config.solver.name} is not supported. Choose from {list(_solvers.keys())}."
            )
        
        optimizer = _solvers[self.config.solver.name](
            self.parameters(),
            lr=self.config.solver.learning_rate,
            weight_decay=self.config.solver.weight_decay if self.config.solver.weight_decay else 0.0
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=lambda curr_epoch: warmup_cosine_lr_lambda(
                        curr_epoch,
                        warmup_epochs=self.config.solver.warmup_epochs,
                        max_epochs=self.config.solver.max_epochs
                    )
                ),
                "interval": "epoch",  # Update every epoch
                "frequency": 1
            }
        }