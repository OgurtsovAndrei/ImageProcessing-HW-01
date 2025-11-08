from typing import Optional
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision.models import Swin_T_Weights, Swin_S_Weights, Swin_B_Weights  # type: ignore
from .freeze_utils import FreezeStrategy, apply_freeze


def _get_default_weights(pretrained: bool):
    if pretrained:
        return Swin_T_Weights.DEFAULT
    return None

def load_swin_from_weights(weights_path: str, num_classes: int) -> nn.Module:
    # create_swin_classifier returns a single nn.Module; do not try to unpack
    model = create_swin_classifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    return model

def create_swin_classifier(
        num_classes: int = 1000,
        pretrained: bool = True,
        freeze: Optional[FreezeStrategy] = None,
        weights: Optional[object] = None,
) -> nn.Module:
    used_weights = weights if weights is not None else _get_default_weights(pretrained)

    model = models.swin_t(weights=used_weights)

    in_features = model.head.in_features
    if num_classes != model.head.out_features:
        model.head = nn.Linear(in_features, num_classes)
        nn.init.trunc_normal_(model.head.weight, std=0.02)
        if model.head.bias is not None:
            nn.init.zeros_(model.head.bias)

    apply_freeze(model, classifier_prefixes=("head",), strategy=freeze)
    return model


if __name__ == '__main__':

    image_path = "../../data/mac-merged/0.png"
    image = Image.open(image_path).convert('RGB')

    model = create_swin_classifier(num_classes=2, pretrained=True, freeze=FreezeStrategy.LAST)
    model.eval()

    weights = Swin_T_Weights.DEFAULT
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
