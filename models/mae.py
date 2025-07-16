import torch
import timm
from torch import nn
from einops import rearrange

from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM


class MAE(nn.Module):
    def __init__(
            self,
            img_size: int,
            in_chans: int = 3,
            num_frames: int = 1,
            embed_dim: int = 768,
            num_heads: int = 8,
            ) -> None:
        super().__init__()

        decoder_dim = 512
        self.mask_ratio = 0.75
        self.patch_size = 16
        self.in_chans = in_chans

        self.encoder = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            img_size=img_size,
            in_chans=in_chans,
            num_classes=0,
        )

        self.backbone = MaskedVisionTransformerTIMM(vit=self.encoder)
        self.sequence_length = self.backbone.sequence_length
        self.decoder = MAEDecoderTIMM(
            num_patches=self.encoder.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=self.encoder.embed_dim,
            decoder_embed_dim=decoder_dim,
            decoder_depth=1,
            decoder_num_heads=16,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
        )
        self.decoder_pred = nn.Linear(decoder_dim, self.patch_size * self.patch_size * in_chans)

    def forward_encoder(self, images: torch.Tensor, idx_keep: list[int] | None = None) -> torch.Tensor:
        return self.backbone.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(
            self,
            x_encoded: torch.Tensor,
            idx_keep: list[int]
        ) -> torch.Tensor:
        # build decoder input
        batch_size = x_encoded.shape[0]

        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.decoder.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for all tokens
        x_pred = self.decoder_pred(x_decoded)
        
        return x_pred

    def forward(
            self,
            images: torch.Tensor,
            temporal_coords: None | torch.Tensor = None,
            mask_ratio: None | float = None
        ) -> dict[str, torch.Tensor]:
        batch_size = images.shape[0]

        mask_ratio = mask_ratio if mask_ratio is not None else self.mask_ratio

        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
        x_pred = self.forward_decoder(x_encoded=x_encoded, idx_keep=idx_keep)

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)

        loss = nn.functional.mse_loss(
            utils.get_at_index(x_pred, idx_mask - 1),
            utils.get_at_index(patches, idx_mask - 1)
        )

        x_pred = x_pred[:, 1:, :]  # Remove class token
        x_pred = self.unpatchify(x_pred, img_size=images.shape[-2:])

        return {
            "inputs": images,
            "loss": loss,
            "preds": x_pred,
            "masks": idx_mask,
            "latents": x_encoded
        }
    
    def unpatchify(self, patches, img_size: tuple[int, int]):
        """
        Args:
            patches: Tensor of shape (B, N, patch_dim)
            patch_size: int, e.g., 16
            img_size: tuple (H, W)
            in_chans: number of input channels
        Returns:
            images: Tensor of shape (B, in_chans, H, W)
        """
        _, N, patch_dim = patches.shape
        H, W = img_size
        assert patch_dim == self.patch_size * self.patch_size * self.in_chans

        h, w = H // self.patch_size, W // self.patch_size  # number of patches per row/colw, N)
        assert N == h * w, "Mismatch between number of patches and image size"

        return rearrange(
            patches,
            'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            h=h, w=w, p1=self.patch_size, p2=self.patch_size, c=self.in_chans
        )


# if __name__ == "__main__":
#     model = MAE(
#         img_size=96,
#         in_chans=10,
#         num_frames=1
#     )
    
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.to(device)
    
#     x = torch.randn(2, 10, 96, 96).to(device)
    
#     outputs = model(x)
    
#     print(outputs["latents"].shape)
#     print(outputs["preds"].shape, outputs["inputs"].shape)