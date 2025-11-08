from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision.models import ViT_B_16_Weights  # type: ignore

from src.model.freeze_utils import FreezeStrategy, apply_freeze


def _get_default_weights(pretrained: bool):
    if pretrained:
        return ViT_B_16_Weights.DEFAULT
    return None

def load_vit_from_weights(weights_path: str, num_classes: int) -> nn.Module:
    model = create_vit_classifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    return model

def create_vit_classifier(
        num_classes: int = 1000,
        pretrained: bool = True,
        freeze: Optional[FreezeStrategy] = None,
        weights: Optional[object] = None,
) -> nn.Module:
    used_weights = weights if weights is not None else _get_default_weights(pretrained)

    model = models.vit_b_16(weights=used_weights)

    in_features = model.heads.head.in_features
    if num_classes != model.heads.head.out_features:
        model.heads.head = nn.Linear(in_features, num_classes)
        nn.init.trunc_normal_(model.heads.head.weight, std=0.02)
        if model.heads.head.bias is not None:
            nn.init.zeros_(model.heads.head.bias)

    apply_freeze(model, classifier_prefixes=("heads.head",), strategy=freeze)
    return model

def load_regular_vit():
    image_path = "../../data/mac-merged/0.png"
    image = Image.open(image_path).convert('RGB')

    model = create_vit_classifier(num_classes=2, pretrained=True, freeze=FreezeStrategy.LAST)
    model.eval()

    weights = ViT_B_16_Weights.DEFAULT
    preprocess = weights.transforms()

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class].item()

    print(f"Image: {image_path}")
    print(f"Image size: {image.size}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Class probabilities: {probabilities.tolist()}")


if __name__ == '__main__':
    load_regular_vit()