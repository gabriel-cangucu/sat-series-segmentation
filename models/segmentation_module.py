import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from typing import Any
from segmentation_models_pytorch.losses import DiceLoss

from utils import get_preds_from_logits
from .utils.necks import SampleFeatures, ReshapeFeatures
from .backbones.mae import MAE
from .backbones.prithvi_mae import PrithviMAE
from .backbones.swin_mae import SwinMAE
from .decoders.swin_unet_decoder import SwinUnetDecoder
from .decoders.segformer_decoder import SegformerDecoder

_backbones = {
    "mae": MAE,
    "prithvi_mae": PrithviMAE,
    "swin_mae": SwinMAE
}

_decoders = {
    "segformer": SegformerDecoder,
    "swin_unet": SwinUnetDecoder,
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
        self.num_timestamps = config.model.num_frames
        self.mode = "binary" if config.model.num_classes == 1 else "multiclass"
        self.ignore_index = config.solver.ignore_index if hasattr(config.solver, "ignore_index") else None
        
        if config.model.backbone not in _backbones:
            raise ValueError(
                f"Unsupported model type: {config.model.backbone}. Choose from: {list(_backbones.keys())}."
            )
        
        backbone = _backbones[config.model.backbone]
        self.backbone = backbone(
            img_size=config.model.img_size,
            patch_size=config.model.patch_size,
            in_chans=config.model.num_channels,
            num_frames=config.model.num_frames
        )

        self.neck = nn.Sequential(*[
            SampleFeatures(indices=[2, 5, 8, 11]),
            ReshapeFeatures()
        ])

        decoder = _decoders[config.model.decoder]
        self.decoder = decoder(
            img_size=config.model.img_size,
            num_classes=config.model.num_classes,
            embed_dims=[768, 384, 192, 96] if config.model.backbone == "swin_mae" else [768] * 4
        )

        if config.solver.criterion not in _losses:
            raise ValueError(
                f"Unsupported loss function: {config.solver.criterion}. Choose from: {list(_losses.keys())}."
            )

        loss_args = {
            "cross_entropy": {
                "weight": torch.tensor(config.solver.class_weights) if config.solver.class_weights else None,
                "ignore_index": self.ignore_index if self.ignore_index else -100
            },
            "dice": {
                "mode": self.mode,
                "from_logits": True,
                "log_loss": True,
                "ignore_index": self.ignore_index
            }
        }
        criterion = _losses[config.solver.criterion]
        self.criterion = criterion(**loss_args[config.solver.criterion])

        metrics_args = {
            "task": self.mode,
            "num_classes": None if self.mode == "binary" else self.num_classes,
            "ignore_index": self.ignore_index
        }
        metrics = torchmetrics.MetricCollection({
            "accuracy": torchmetrics.Accuracy(**metrics_args),
            "precision": torchmetrics.Precision(**metrics_args),
            "recall": torchmetrics.Recall(**metrics_args),
            "f1": torchmetrics.F1Score(**metrics_args),
            "iou": torchmetrics.JaccardIndex(**metrics_args)
        })

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self,
        batch: dict[str, torch.Tensor],
        temporal_coords: None | torch.Tensor = None,
    ) -> torch.Tensor:
        x = batch["data"]

        x, features = self.backbone.forward_features(x)
        features = self.neck(features)
        logits = self.decoder(x, features)

        return {
            "inputs": batch["data"],
            "targets": batch["target"],
            "logits": logits
        }

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        outputs = self(batch, temporal_coords=batch["dates"])

        loss = self.criterion(outputs["logits"], batch["target"])
        preds = get_preds_from_logits(outputs["logits"], num_classes=self.num_classes)

        self.train_metrics.update(preds, batch["target"])
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        for i, param_group in enumerate(self.optimizers().param_groups):
            self.log(f"lr/group_{i}", param_group["lr"], on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), prog_bar=True)
        self.train_metrics.reset()
    
    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self(batch, temporal_coords=batch["dates"])

        loss = self.criterion(outputs["logits"], batch["target"])
        preds = get_preds_from_logits(outputs["logits"], num_classes=self.num_classes)

        self.val_metrics.update(preds, batch["target"])
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
    
    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True)
        self.val_metrics.reset()
    
    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self(batch, temporal_coords=batch["dates"])

        loss = self.criterion(outputs["logits"], batch["target"])
        preds = get_preds_from_logits(outputs["logits"], num_classes=self.num_classes)

        self.test_metrics.update(preds, batch["target"])
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
    
    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), prog_bar=True)
        self.test_metrics.reset()
    
    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self(batch, temporal_coords=batch["dates"])

    def configure_optimizers(self) -> dict[str, Any]:
        if self.config.solver.name not in _solvers:
            raise ValueError(
                f"Solver {self.config.solver.name} is not supported. Choose from {list(_solvers.keys())}."
            )
        
        optimizer = _solvers[self.config.solver.name]([
            {
                "params": self.backbone.parameters(),
                "lr": 1e-5,
                "weight_decay": self.config.solver.weight_decay if self.config.solver.weight_decay else 0.0
            },
            {
                "params": self.decoder.parameters(),
                "lr": self.config.solver.learning_rate,
                "weight_decay": self.config.solver.weight_decay if self.config.solver.weight_decay else 0.0
            },
        ])
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode="min", factor=0.1, patience=5
        # )
        
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": scheduler,
            #     "monitor": "val/loss",
            #     "interval": "epoch",
            #     "frequency": 1
            # }
        }
