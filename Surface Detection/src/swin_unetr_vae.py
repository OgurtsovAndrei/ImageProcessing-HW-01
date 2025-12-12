from math import prod

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence
from monai.networks.blocks import UnetOutBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep
from monai.networks.nets import SwinUNETR


class SwinUNETRVAE(SwinUNETR):
    """
    SwinUNETR with VAE regularization.

    Encoder and segmentation decoder architecture are inherited from SwinUNETR.
    A VAE branch is added on top of the encoder bottleneck to reconstruct the input image.
    This adds an additional regularization term encouraging informative latent features.
    """

    def __init__(
            self,
            input_image_size: Sequence[int],
            in_channels: int,
            out_channels: int,
            vae_estimate_std: bool = False,
            vae_default_std: float = 0.3,
            vae_nz: int = 256,
            feature_size: int = 24,
            spatial_dims: int = 3,
            **kwargs
    ) -> None:

        # SwinUNETR expects:
        #   - spatial_dims == 3 for 3D inputs
        #   - feature_size to be a multiple of 12
        #   - input spatial size to be divisible by patch_size ** 5
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            spatial_dims=spatial_dims,
            **kwargs
        )

        # Number of input channels for the original image X
        self.in_channels = in_channels

        # Spatial dimensionality (2D or 3D). This implementation is intended for 3D (D, H, W).
        self.spatial_dims = spatial_dims

        # Expected spatial size of the input patch:
        #   input_image_size ~ (D, H, W) for spatial_dims == 3
        # Must be consistent with what you actually feed into the network.
        self.input_image_size = ensure_tuple_rep(input_image_size, spatial_dims)

        # VAE configuration parameters
        self.vae_estimate_std = vae_estimate_std
        self.vae_default_std = vae_default_std
        self.vae_nz = vae_nz  # latent dimensionality z: shape [B, vae_nz]

        # SwinUNETR downsamples by patch_size ** 5 from input to the encoder bottleneck.
        # For example:
        #   patch_size = 2  =>  scale_factor = 2^5 = 32
        # If input_image_size is (96, 96, 96), bottleneck spatial size will be (3, 3, 3).
        self.scale_factor = self.patch_size ** 5

        # Bottleneck spatial size: [D / scale_factor, H / scale_factor, W / scale_factor]
        # Each dimension must be an integer (input should be divisible by scale_factor).
        self.bottleneck_size = [d // self.scale_factor for d in self.input_image_size]

        # Bottleneck channel count:
        # For SwinUNETR: the last encoder stage uses 16 * feature_size channels.
        # Bottleneck tensor shape: [B, bottleneck_channels, Db, Hb, Wb]
        self.bottleneck_channels = 16 * feature_size

        # Prepare all VAE-specific modules (MLP for mean/std and decoder)
        self._prepare_vae_modules()

    def _prepare_vae_modules(self):
        # Flattened size of the bottleneck:
        #   bottleneck_features.view(B, -1) has size flat_size
        #   flat_size = Cb * Db * Hb * Wb
        self.flat_size = self.bottleneck_channels * prod(self.bottleneck_size)

        # Fully connected layers used to compute mean and std of latent z from bottleneck
        # Input:  [B, flat_size]
        # Output: [B, vae_nz]
        self.vae_fc1 = nn.Linear(self.flat_size, self.vae_nz)  # z_mean
        self.vae_fc2 = nn.Linear(self.flat_size, self.vae_nz)  # z_sigma (if estimated)
        self.vae_fc3 = nn.Linear(self.vae_nz, self.flat_size)  # maps z back to bottleneck shape

        # VAE decoder: starts from bottleneck spatial size and upsamples back
        # to the original image spatial size using 5 transposed-convolution blocks.
        #
        # Shapes (channels only, spatial doubled after each block):
        #   start:  C = bottleneck_channels = 16 * feature_size
        #   block 0: C -> C / 2
        #   block 1: C / 2 -> C / 4
        #   block 2: C / 4 -> C / 8
        #   block 3: C / 8 -> C / 16
        #   block 4: C / 16 -> C / 32
        #
        # After 5 blocks spatial size is scaled by 2^5 = 32,
        # which matches the downsampling factor from input to bottleneck.
        layers = []
        current_ch = self.bottleneck_channels

        for i in range(5):
            out_ch = current_ch // 2 if i < 4 else current_ch // 2

            # For 3D:
            #   input:  [B, C, D_i, H_i, W_i]
            #   output: [B, out_ch, 2*D_i, 2*H_i, 2*W_i]
            # For 2D (if ever used):
            #   input:  [B, C, H_i, W_i]
            #   output: [B, out_ch, 2*H_i, 2*W_i]
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        current_ch,
                        out_ch,
                        kernel_size=2,
                        stride=2,
                    )
                    if self.spatial_dims == 3
                    else nn.ConvTranspose2d(
                        current_ch,
                        out_ch,
                        kernel_size=2,
                        stride=2,
                    ),
                    nn.InstanceNorm3d(out_ch)
                    if self.spatial_dims == 3
                    else nn.InstanceNorm2d(out_ch),
                    nn.LeakyReLU(inplace=True),
                )
            )
            current_ch = out_ch

        # Sequential stack of 5 upsampling blocks
        self.vae_decoder = nn.Sequential(*layers)

        # Final 1x1 convolution to map decoder channels back to original input channels.
        # For 3D:  [B, C_dec, D, H, W] -> [B, in_channels, D, H, W]
        self.vae_conv_final = (
            nn.Conv3d(current_ch, self.in_channels, kernel_size=1)
            if self.spatial_dims == 3
            else nn.Conv2d(current_ch, self.in_channels, kernel_size=1)
        )

    def _get_vae_loss(self, net_input: torch.Tensor, bottleneck_features: torch.Tensor):
        """
        Compute VAE loss (reconstruction + KL-like regularization).

        Args:
            net_input:
                Original network input X.
                Shape for 3D: [B, C_in, D, H, W]
            bottleneck_features:
                Encoder bottleneck feature map taken from dec4.
                Shape for 3D: [B, bottleneck_channels, Db, Hb, Wb]

        Returns:
            Scalar VAE loss = reconstruction MSE + regularization term.
        """
        # Batch size B
        b_size = bottleneck_features.shape[0]

        # Flatten bottleneck features:
        #   bottleneck_features: [B, Cb, Db, Hb, Wb]
        #   x_flat:              [B, Cb * Db * Hb * Wb] = [B, flat_size]
        x_flat = bottleneck_features.view(b_size, -1)

        # Latent mean: z_mean shape [B, vae_nz]
        z_mean = self.vae_fc1(x_flat)

        if self.vae_estimate_std:
            # Latent std parameterization:
            #   raw sigma: [B, vae_nz]
            #   softplus to ensure positivity
            z_sigma = self.vae_fc2(x_flat)
            z_sigma = F.softplus(z_sigma)

            # Regularization term: KL divergence between N(z_mean, z_sigma^2) and N(0, 1)
            #   vae_reg_loss ~ mean over batch and latent dims
            vae_reg_loss = 0.5 * torch.mean(
                z_mean ** 2 + z_sigma ** 2 - torch.log(1e-8 + z_sigma ** 2) - 1
            )

            # Reparameterization trick:
            #   z = z_mean + z_sigma * epsilon
            #   epsilon ~ N(0, I)
            z_mean_rand = torch.randn_like(z_mean)
            x_vae = z_mean + z_sigma * z_mean_rand
        else:
            # Fixed std for latent space: z ~ N(z_mean, I * vae_default_std^2)
            # Regularization is L2 on z_mean (pushes it towards 0).
            z_sigma = self.vae_default_std
            vae_reg_loss = torch.mean(z_mean ** 2)
            z_mean_rand = torch.randn_like(z_mean)
            x_vae = z_mean + z_sigma * z_mean_rand

        # Map latent vector z back to flattened bottleneck shape:
        #   x_vae: [B, vae_nz] -> [B, flat_size]
        x_vae = self.vae_fc3(x_vae)
        x_vae = F.leaky_relu(x_vae)

        # Reshape to bottleneck tensor:
        #   [B, flat_size] -> [B, Cb, Db, Hb, Wb]
        x_vae = x_vae.view(b_size, self.bottleneck_channels, *self.bottleneck_size)

        # Decode through transposed-convolution blocks:
        #   x_rec shape after decoder:
        #     3D: [B, C_dec, D, H, W] (after 5× upsampling)
        x_rec = self.vae_decoder(x_vae)

        # Final conv to reconstruct original channels:
        #   [B, C_dec, D, H, W] -> [B, C_in, D, H, W]
        x_rec = self.vae_conv_final(x_rec)

        # Spatial size safety check:
        # due to integer divisions or minor shape mismatches,
        # we ensure x_rec spatial size matches net_input.
        if x_rec.shape != net_input.shape:
            x_rec = F.interpolate(
                x_rec,
                size=net_input.shape[2:],  # (D, H, W) for 3D
                mode="trilinear" if self.spatial_dims == 3 else "bilinear",
                align_corners=False,
            )

        # Reconstruction loss (MSE in input space):
        #   net_input: [B, C_in, D, H, W]
        #   x_rec:     [B, C_in, D, H, W]
        vae_mse_loss = F.mse_loss(net_input, x_rec)

        # Total VAE loss = regularization + reconstruction error
        vae_loss = vae_reg_loss + vae_mse_loss
        return vae_loss

    def forward(self, x_in: torch.Tensor):
        """
        Forward pass through SwinUNETR segmentation path + VAE branch.

        Args:
            x_in:
                Input volume.
                Shape for 3D: [B, C_in, D, H, W]

        Returns:
            logits:
                Segmentation logits.
                Shape: [B, out_channels, D, H, W]
            vae_loss:
                Scalar VAE loss (or None in eval mode).
        """
        # Ensure input size is compatible with SwinUNETR requirements:
        # spatial dims must be divisible by patch_size ** 5.
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._check_input_size(x_in.shape[2:])

        # Pass input through Swin ViT encoder backbone.
        # hidden_states_out is a list of feature maps at different resolutions.
        # Typical shapes (3D case, depends on config):
        #   hidden_states_out[0]: [B, C1, D/2,   H/2,   W/2  ]
        #   hidden_states_out[1]: [B, C2, D/4,   H/4,   W/4  ]
        #   hidden_states_out[2]: [B, C3, D/8,   H/8,   W/8  ]
        #   hidden_states_out[3]: [B, C4, D/16,  H/16,  W/16 ]
        #   hidden_states_out[4]: [B, C5, D/32,  H/32,  W/32 ]
        hidden_states_out = self.swinViT(x_in, self.normalize)

        # UNETR-style encoder path using Swin features as skip connections.
        # enc0: shallow conv on input
        #   [B, C_in, D, H, W] -> [B, feature_size, D, H, W]
        enc0 = self.encoder1(x_in)

        # enc1:
        #   hidden_states_out[0]: [B, C1, D/2, H/2, W/2]
        #   enc1:                 [B, feature_size * 2, D/2, H/2, W/2]
        enc1 = self.encoder2(hidden_states_out[0])

        # enc2:
        #   hidden_states_out[1]: [B, C2, D/4, H/4, W/4]
        #   enc2:                 [B, feature_size * 4, D/4, H/4, W/4]
        enc2 = self.encoder3(hidden_states_out[1])

        # enc3:
        #   hidden_states_out[2]: [B, C3, D/8, H/8, W/8]
        #   enc3:                 [B, feature_size * 8, D/8, H/8, W/8]
        enc3 = self.encoder4(hidden_states_out[2])

        # dec4 (bottleneck):
        #   hidden_states_out[4]: [B, C5, D/32, H/32, W/32]
        #   dec4:                 [B, 16 * feature_size, D/32, H/32, W/32]
        # This is the feature map we use as VAE bottleneck.
        dec4 = self.encoder10(hidden_states_out[4])

        # Decoder path with skip connections (standard SwinUNETR behavior).
        # dec3:
        #   input:  dec4 + hidden_states_out[3]
        #   shape:  approx [B, 8 * feature_size, D/16, H/16, W/16]
        dec3 = self.decoder5(dec4, hidden_states_out[3])

        # dec2:
        #   input:  dec3 + enc3
        #   shape:  approx [B, 4 * feature_size, D/8, H/8, W/8]
        dec2 = self.decoder4(dec3, enc3)

        # dec1:
        #   input:  dec2 + enc2
        #   shape:  approx [B, 2 * feature_size, D/4, H/4, W/4]
        dec1 = self.decoder3(dec2, enc2)

        # dec0:
        #   input:  dec1 + enc1
        #   shape:  approx [B, feature_size, D/2, H/2, W/2]
        dec0 = self.decoder2(dec1, enc1)

        # out:
        #   input:  dec0 + enc0
        #   shape:  approx [B, feature_size, D, H, W]
        out = self.decoder1(dec0, enc0)

        # Final segmentation logits:
        #   [B, feature_size, D, H, W] -> [B, out_channels, D, H, W]
        logits = self.out(out)

        # VAE loss is computed only in training mode to avoid extra cost at inference.
        if self.training:
            # dec4 is used as bottleneck feature map for VAE.
            vae_loss = self._get_vae_loss(x_in, dec4)
        else:
            vae_loss = None

        # Forward always returns (logits, vae_loss) to be compatible
        # with SegResNetVAE-style usage in ensembles.
        return logits, vae_loss
