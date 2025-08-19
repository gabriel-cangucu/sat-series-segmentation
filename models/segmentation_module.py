import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from typing import Any
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.losses import DiceLoss

from .mae import MAE
from .prithvi_mae import PrithviMAE

_models = {
    "mae": MAE,
    "prithvi_mae": PrithviMAE,
}

_losses = {
    "cross_entropy": nn.CrossEntropyLoss,
    "dice": DiceLoss
}

_solvers = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW
}


class SegmentationModule(L.LightningModule):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        
        self.config = config
        self.save_hyperparameters()
        
        self.num_classes = config.model.num_classes
        self.mode = "binary" if config.model.num_classes == 1 else "multiclass"
        self.ignore_index = config.solver.ignore_index if hasattr(config.solver, "ignore_index") else None
        
        if config.model.name not in _models:
            raise ValueError(
                f"Unsupported model type: {config.model.name}. Choose from: {list(_models.keys())}."
            )
        
        net = _models[config.model.name]
        self.net = net(
            img_size=config.model.img_size,
            in_chans=config.model.num_channels,
            num_frames=config.model.num_timestamps
        )
        self.encoder = self.net.encoder
        self.segmentation_head = SegmentationHead(
            in_channels=self.net.embed_dim,
            out_channels=config.model.num_classes,
            activation=None,
            kernel_size=3
        )
        
        if config.solver.criterion not in _losses:
            raise ValueError(
                f"Unsupported loss function: {config.solver.criterion}. Choose from: {list(_losses.keys())}."
            )
        
        criterion = _losses[config.solver.criterion]
        # Does not work with CrossEntropyLoss yet!
        self.criterion = criterion(
            mode=self.mode,
            from_logits=True,
            log_loss=True,
            ignore_index=self.ignore_index
        )
        
        self.accuracy = torchmetrics.Accuracy(
            task=self.mode, num_classes=self.num_classes, ignore_index=self.ignore_index
        )
        self.miou = torchmetrics.JaccardIndex(
            task=self.mode, num_classes=self.num_classes, ignore_index=self.ignore_index
        )
    
    def forward(self,
        x: torch.Tensor,
        temporal_coords: None | torch.Tensor = None,
    ) -> torch.Tensor:
        latent, _, _ = self.encoder(x, temporal_coords, location_coords=None, mask_ratio=0.)
        return self.segmentation_head(latent)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        outputs = self(batch["data"], temporal_coords=batch["dates"])
        
        loss = self.criterion(outputs, batch["target"])
        accuracy = self.accuracy(outputs, batch["target"])
        miou = self.miou(outputs, batch["target"])
        learning_rate = self.optimizers().param_groups[0]["lr"]
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_miou", miou, on_step=True, on_epoch=True, prog_bar=True)
        self.log("learning_rate", learning_rate, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self(batch["data"], temporal_coords=batch["dates"])
        
        loss = self.criterion(outputs, batch["target"])
        accuracy = self.accuracy(outputs, batch["target"])
        miou = self.miou(outputs, batch["target"])
        
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_miou", miou, on_step=True, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self(batch["data"], temporal_coords=batch["dates"])
        
        loss = self.criterion(outputs, batch["target"])
        accuracy = self.accuracy(outputs, batch["target"])
        miou = self.miou(outputs, batch["target"])
        
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_miou", miou, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> dict[str, Any]:
        if self.config.solver.name not in _solvers:
            raise ValueError(
                f"Solver {self.config.solver.name} is not supported. Choose from {list(_solvers.keys())}."
            )
        
        optimizer = _solvers[self.config.solver.name](
            self.parameters(),
            lr=self.config.solver.learning_rate,
            weight_decay=self.config.solver.weight_decay if self.config.solver.weight_decay else 0.0
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }
