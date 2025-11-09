import torch
import os

from huggingface_hub import hf_hub_download
from monai.networks.nets import SwinUNETR
from networkx.classes import freeze
from torchinfo import summary
import segmentation_models_pytorch as smp
import freeze_utils


def adapt_radimagenet_weights(device):
    """
    Загружает веса RadImageNet-InceptionResNetV2, адаптирует
    первый слой (3 канала -> 1 канал) и возвращает state_dict.
    """
    print("Downloading RadImageNet InceptionResNetV2 weights...")
    weights_path = hf_hub_download(
        repo_id="Lab-Rasool/RadImageNet",
        filename="InceptionResNetV2.pt"  # <--- ИЗМЕНЕНО
    )

    weights = torch.load(weights_path, map_location=torch.device('cpu'))

    # --- Адаптация 1-го слоя ---
    # У InceptionResNetV2 первый слой называется 'conv2d_1a.conv.weight'
    conv1_weights_3ch = weights['conv2d_1a.conv.weight']

    # "Схлопываем" 3 канала в 1
    conv1_weights_1ch = conv1_weights_3ch.sum(dim=1, keepdim=True)

    # Заменяем веса в словаре
    weights['conv2d_1a.conv.weight'] = conv1_weights_1ch
    print("Weights adapted for 1-channel input.")

    return weights


def create_model(device, num_classes=4):
    """
    Создает 2D Unet с энкодером InceptionResNetV2, загружает веса RadImageNet
    и адаптирует их для 1-канального входа.
    """

    print("Creating 2D Unet model with inceptionresnetv2 backbone...")

    model = smp.Unet(
        encoder_name="inceptionresnetv2",
        encoder_weights=None,
        in_channels=1,
        classes=num_classes,
    )

    try:
        adapted_weights = adapt_radimagenet_weights(device)
        model.encoder.load_state_dict(adapted_weights, strict=False)
        print("Successfully loaded RadImageNet weights into 1-channel encoder.")

    except Exception as e:
        print(f"Error loading RadImageNet weights: {e}")
        print("Proceeding with randomly initialized weights.")

    model.to(device)
    return model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(device, num_classes=4)

    freeze_utils.apply_freeze(model, ("SegmentationHead",), strategy=freeze_utils.FreezeStrategy.PCT70)

    print("\nModel created successfully:")

    # --- Печать статистики ---
    batch_size = 8
    input_channels = 1
    image_size = 256

    print("\n--- Model Summary (Unet + RadImageNet-ResNet50) ---")
    summary(model, input_size=(batch_size, input_channels, image_size, image_size))