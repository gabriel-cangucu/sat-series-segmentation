import torch
import numpy as np
import lightning as L
from typing import Any
from terratorch.models.backbones import prithvi_mae
from lightly.utils.debug import std_of_l2_normalized

from utils import get_random_embedding

_solvers = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW
}


class terratorch_PrithviMAE(prithvi_mae.PrithviMAE):
    def __init__(
        self,
        img_size: int,
        in_chans: int,
        num_frames: int,
        embed_dim: int,
        num_heads: int,
        coords_encoding: list[str] = ["time"],
        coords_scale_learn: bool = True
    ) -> None:
        super().__init__(
            img_size=img_size,
            in_chans=in_chans,
            num_frames=num_frames,
            embed_dim=embed_dim,
            num_heads=num_heads,
            coords_encoding=coords_encoding,
            coords_scale_learn=coords_scale_learn
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        temporal_coords: None | torch.Tensor = None,
        location_coords: None | torch.Tensor = None,
        mask_ratio: None | float = None,
    ) -> dict[str, torch.Tensor]:
        if len(pixel_values.shape) == 4 and self.encoder.patch_embed.input_size[0] == 1:
            # add time dim
            pixel_values = pixel_values.unsqueeze(2)
            time_dim_added = True
        else:
            time_dim_added = False

        mask_ratio = mask_ratio if mask_ratio is not None else self.mask_ratio

        latent, mask, ids_restore = self.encoder(pixel_values, temporal_coords, location_coords, mask_ratio)
        pred = self.decoder(latent, ids_restore, temporal_coords, location_coords, input_size=pixel_values.shape)
        loss = self.forward_loss(pixel_values, pred, mask)

        # Prepare output format in TerraTorch
        mask = mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1])
        mask = self.unpatchify(mask, image_size=pixel_values.shape[-2:])
        mask = mask[:, 0] # Remove channel dim
        pred = self.unpatchify(pred, image_size=pixel_values.shape[-2:])
        
        if time_dim_added:
            # Remove time dim to match input data
            pred = pred.squeeze(2)
            mask = mask.squeeze(1)
        
        return {
            "inputs": pixel_values,
            "loss": loss,
            "preds": pred,
            "masks": mask,
            "latents": latent
        }


class PrithviMAE(L.LightningModule):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        
        self.config = config
        
        self.save_hyperparameters(config)
        self.net = terratorch_PrithviMAE(
            img_size=config.model.img_size,
            in_chans=config.model.num_channels,
            num_frames=config.model.num_timestamps,
            embed_dim=768,
            num_heads=8
        )

    def forward(self,
        x: torch.Tensor,
        temporal_coords: None | torch.Tensor = None,
        mask_ratio: float = None
    ) -> torch.Tensor:
        return self.net(x, temporal_coords=temporal_coords, mask_ratio=mask_ratio)
    
    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        outputs = self(batch["data"], temporal_coords=batch["dates"])
        
        latent = get_random_embedding(outputs["latents"])
        embeddings_std = torch.nan_to_num(std_of_l2_normalized(latent))
        
        self.log("train_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("embeddings_std", embeddings_std, on_step=True, on_epoch=True, prog_bar=True)
        
        return outputs["loss"]
    
    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        outputs = self(batch["data"], temporal_coords=batch["dates"])
        
        self.log("test_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True)
        
        return outputs["loss"]
    
    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        return self(batch["data"], temporal_coords=batch["dates"])
    
    def configure_optimizers(self):
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
