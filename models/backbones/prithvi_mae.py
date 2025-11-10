import torch
from terratorch.models.backbones import prithvi_mae


class PrithviMAE(prithvi_mae.PrithviMAE):
    def __init__(
            self,
            img_size: int = 224,
            in_chans: int = 3,
            num_frames: int = 2,
            patch_size: int = 16,
        ) -> None:
        self.patch_size = (num_frames, patch_size, patch_size)
        
        super().__init__(
            img_size=img_size,
            in_chans=in_chans,
            num_frames=num_frames,
            patch_size=self.patch_size
        )
    
    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Returns a list of transformer block features
        features = self.encoder.forward_features(x)
        return x, features
    
    def forward_encoder(
            self, x: torch.Tensor,
            mask_ratio: float
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encoder.forward(x, mask_ratio=mask_ratio)

    def forward(
            self,
            x: torch.Tensor,
            temporal_coords: None | torch.Tensor = None,
            location_coords: None | torch.Tensor = None,
            mask_ratio: None | float = 0.75,
        ) -> dict[str, torch.Tensor]:
        if len(x.shape) == 4 and self.encoder.patch_embed.input_size[0] == 1:
            # add time dim
            x = x.unsqueeze(2)
            time_dim_added = True
        else:
            time_dim_added = False

        mask_ratio = mask_ratio if mask_ratio is not None else self.mask_ratio

        latent, mask, ids_restore = self.encoder(x, temporal_coords, location_coords, mask_ratio)
        pred = self.decoder(latent, ids_restore, temporal_coords, location_coords, input_size=x.shape)
        loss = self.forward_loss(x, pred, mask)

        # Prepare output format in TerraTorch
        mask = mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1])
        mask = self.unpatchify(mask, image_size=x.shape[-2:])
        mask = mask[:, 0] # Remove channel dim
        pred = self.unpatchify(pred, image_size=x.shape[-2:])
        
        if time_dim_added:
            # Remove time dim to match input data
            pred = pred.squeeze(2)
            mask = mask.squeeze(1)
        
        return {
            "inputs": x,
            "loss": loss,
            "preds": pred,
            "masks": mask,
            "latents": latent
        }


if __name__ == "__main__":
    model = PrithviMAE(
        img_size=96,
        patch_size=8,
        in_chans=10,
        num_frames=2
    )
    
    x = torch.randn(16, 10, 2, 96, 96)
    outputs = model(x)
    
    print(outputs["preds"].shape)
    
    features = model.forward_features(x)
    for f in features:
        print(f.shape)