import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from transformers.modeling_outputs import ImageClassifierOutput
from typing import List, Union


class DinoV3Classifier(nn.Module):

    def __init__(self, model_name: str, num_labels: int):
        super(DinoV3Classifier, self).__init__()
        self.num_labels = num_labels
        # self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name)

        in_features = self.base_model.config.hidden_size

        self.classifier_head = nn.Linear(in_features, num_labels)

    def forward(self, pixel_values: torch.Tensor):

        # device = self.classifier_head.weight.device

        outputs = self.base_model(pixel_values=pixel_values)
        image_features = outputs.pooler_output

        logits = self.classifier_head(image_features)

        return logits

def create_dino_vit_classifier(
        num_classes: int = 1000,
        freeze_backbone: bool = False,
        model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
) -> DinoV3Classifier:
    model = DinoV3Classifier(model_name, num_classes)

    if freeze_backbone:
        for param in model.base_model.parameters():
            param.requires_grad = False

    return model


def load_dinoV3_vit():

    image_path = "data/mac-merged/0.png"
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        print("Please update the 'image_path' variable to a valid file.")
        return

    num_labels = 2

    model = create_dino_vit_classifier(num_classes=num_labels)
    model.eval()

    with torch.no_grad():
        outputs = model(images=image)
        logits = outputs.logits

    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class_idx = logits.argmax(-1).item()

    print(f"--- Model Structure ---")
    print(model.classifier_head)
    print(f"\n--- Inference Results (Random) ---")
    print(f"Logits: {logits.numpy()}")
    print(f"Probabilities: {probabilities.numpy()}")
    print(f"Predicted class: {predicted_class_idx}")


if __name__ == "__main__":
    load_dinoV3_vit()