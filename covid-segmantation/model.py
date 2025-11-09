import torch
import os
from monai.networks.nets import SwinUNETR


def create_model(device, num_classes=4, pretrained_weights_path="models/model_swinvit.pt"):
    """
    Создает SOTA 3D-модель (SwinUNETR) для сегментации и загружает
    предобученные веса для fine-tuning.

    Args:
        device (torch.device): Устройство (GPU/CPU), на котором будет модель.
        num_classes (int): Количество выходных классов (в вашем случае 4).
        pretrained_weights_path (str): Путь к файлу весов 'model_swinvit.pt'.

    Returns:
        torch.nn.Module: Готовая к обучению SOTA-модель.
    """

    model = SwinUNETR(
        in_channels=1,  # 1 входной канал (КТ)
        out_channels=num_classes,  # Количество классов (как в вашем baseline)
        feature_size=48,  # "Base" версия, соответствует весам model_swinvit.pt [1]
        use_checkpoint=True  # Экономия VRAM (gradient checkpointing)
    )

    if os.path.exists(pretrained_weights_path):
        print(f"Loading model weights: {pretrained_weights_path}...")
        weight = torch.load(pretrained_weights_path, weights_only=True, map_location=torch.device('cpu'))
        model.load_from(weights=weight)

    else:
        raise ValueError()

    model.to(device)

    return model

if __name__ == '__main__':
    create_model(torch.device("cpu"), num_classes=2)