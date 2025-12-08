import torch.nn as nn
from monai.networks.nets import BasicUNet, SegResNet, SwinUNETR, UNETR, SegResNetVAE, DynUNet
from config import IN_CHANNELS, OUT_CHANNELS, PATCH_SIZE


def get_model(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function to initialize and return a neural network model.
    
    Args:
        model_name (str): Name of the model architecture ("BasicUNet" or "SegResNet").
        **kwargs: Additional arguments to pass to the model constructor.
        
    Returns:
        nn.Module: The initialized model.
    """
    if model_name == "BasicUNet":
        return BasicUNet(
            spatial_dims=3,
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            features=(4, 8, 16, 32, 64, 128),
            dropout=0.2,
            **kwargs
        )
    elif model_name == "SwinUNETR":
        return SwinUNETR(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            feature_size=48,
            use_checkpoint=True,
            spatial_dims=3,
            drop_rate=0.05,
            attn_drop_rate=0.05,
            **kwargs
        )
    elif model_name == "UNETR":
        return UNETR(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            img_size=PATCH_SIZE,
            feature_size=32,
            spatial_dims=3,
            dropout_rate=0.05,
            **kwargs
        )
    elif model_name == "SegResNetVAE":
        return SegResNetVAE(
            input_image_size=PATCH_SIZE,
            spatial_dims=3,
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            init_filters=32,
            dropout_prob=0.1,
            vae_nz=128,
            act="RELU",
            norm="INSTANCE",
            upsample_mode="deconv",
        )
    elif model_name == "DynUNet":
        return DynUNet(
            spatial_dims=3,
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            upsample_kernel_size=[2, 2, 2, 2, 2, 2],
            filters=[24, 48, 96, 192, 384],
            strides=[1, 2, 2, 2, 2],
            kernel_size=[3, 3, 3, 3, 3],
            dropout=0.1,
            **kwargs
        )
    elif model_name == "SegResNet":
        return SegResNet(
            spatial_dims=3,
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            init_filters=16,
            dropout_prob=0.2,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}. Available: BasicUNet, SegResNet")
