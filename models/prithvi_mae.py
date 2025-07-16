import torch
from terratorch.models.backbones import prithvi_mae


class PrithviMAE(prithvi_mae.PrithviMAE):
    def __init__(
        self,
        img_size: int,
        in_chans: int,
        num_frames: int,
        embed_dim: int = 768,
        num_heads: int = 8,
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
        images: torch.Tensor,
        temporal_coords: None | torch.Tensor = None,
        location_coords: None | torch.Tensor = None,
        mask_ratio: None | float = None,
    ) -> dict[str, torch.Tensor]:
        if len(images.shape) == 4 and self.encoder.patch_embed.input_size[0] == 1:
            # add time dim
            images = images.unsqueeze(2)
            time_dim_added = True
        else:
            time_dim_added = False

        mask_ratio = mask_ratio if mask_ratio is not None else self.mask_ratio

        latent, mask, ids_restore = self.encoder(images, temporal_coords, location_coords, mask_ratio)
        pred = self.decoder(latent, ids_restore, temporal_coords, location_coords, input_size=images.shape)
        loss = self.forward_loss(images, pred, mask)

        # Prepare output format in TerraTorch
        mask = mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1])
        mask = self.unpatchify(mask, image_size=images.shape[-2:])
        mask = mask[:, 0] # Remove channel dim
        pred = self.unpatchify(pred, image_size=images.shape[-2:])
        
        if time_dim_added:
            # Remove time dim to match input data
            pred = pred.squeeze(2)
            mask = mask.squeeze(1)
        
        return {
            "inputs": images,
            "loss": loss,
            "preds": pred,
            "masks": mask,
            "latents": latent
        }
