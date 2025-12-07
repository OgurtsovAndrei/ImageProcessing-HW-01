import torch.nn as nn
from monai.networks.nets import BasicUNet, SegResNet, SwinUNETR
from config import IN_CHANNELS, OUT_CHANNELS


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
            drop_rate=0.03,
            attn_drop_rate=0.03,
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
