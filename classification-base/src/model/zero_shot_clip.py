import torch
import torch.nn as nn
from typing import List, Tuple
from transformers import CLIPModel, CLIPTokenizerFast

from src.data.dataset import get_STL_dataset, setup_dataset_realtime
from src.model.clip_finetune import calculate_accuracy


def get_module_device(module: nn.Module) -> torch.device:
    try:
        return torch.device("mps")
    except StopIteration:
        return torch.device("cpu")


def freeze_params(module: nn.Module, freeze_top_percent: float = 1.0) -> None:
    all_params_length = len(list(module.parameters()))
    if all_params_length == 0:
        return

    for indx, param in enumerate(module.parameters()):
        if int(all_params_length * freeze_top_percent) <= indx:
            break
        param.requires_grad = False


def print_trainable_parameters(model: nn.Module) -> None:
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    if all_param == 0:
        print("Model has no parameters.")
        return

    print(
        f"Trainable params: {(trainable_params / 10 ** 6):.4f}M || "
        f"All params: {(all_param / 10 ** 6):.4f}M || "
        f"Trainable %: {100 * trainable_params / all_param:.2f}%"
    )


class CLIPClassifier(nn.Module):
    def __init__(self, clip_model: CLIPModel, tokenizer: CLIPTokenizerFast, labels: list[str]):
        super().__init__()
        self.model = clip_model
        self.tokenizer = tokenizer
        self.logit_scale = self.model.logit_scale.exp()
        self.label2id = {label: i for i, label in enumerate(labels)}

        self.labels_embeddings = nn.Parameter(
            self.generate_labels_embeddings(labels, get_module_device(self.model))
        )
        self.labels_embeddings.requires_grad = False

    @torch.no_grad()
    def generate_labels_embeddings(self, labels: list[str], device: torch.device) -> torch.Tensor:
        labels_inputs = self.tokenizer(
            [f"a photo of {label}" for label in labels],
            return_tensors="pt",
            padding=True,
        ).to(device)

        labels_embeddings = self.model.get_text_features(**labels_inputs)

        labels_embeddings /= labels_embeddings.norm(p=2, dim=-1, keepdim=True)
        return labels_embeddings

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.model.get_image_features(
            images
        )  # (batch_size, dim)
        image_features /= image_features.norm(p=2, dim=-1, keepdim=True)

        labels_emb = self.labels_embeddings.to(image_features.device)

        return torch.matmul(image_features, labels_emb.T) * self.logit_scale


def set_up_clip(
        model_name: str,
        classes_labels: List[str],
        tokenizer: CLIPTokenizerFast,
        freeze_text: bool,
        freeze_image: bool,
) -> Tuple[CLIPModel, CLIPClassifier]:
    print(f"Loading base model: {model_name}")
    base_model = CLIPModel.from_pretrained(model_name)

    if freeze_text:
        print("Freezing text model parameters.")
        freeze_params(base_model.text_model)

    if freeze_image:
        print("Freezing vision model parameters.")
        freeze_params(base_model.vision_model)

    model_for_finetuning = base_model

    print("--- Model for Fine-Tuning ---")
    print_trainable_parameters(model_for_finetuning)

    print("\nSetting up zero-shot classifier...")
    classifier_for_zero_shot = CLIPClassifier(base_model, tokenizer, classes_labels)

    freeze_params(classifier_for_zero_shot)

    print("--- Classifier for Zero-Shot (Frozen) ---")
    print_trainable_parameters(classifier_for_zero_shot)

    return model_for_finetuning, classifier_for_zero_shot


# if __name__ == "__main__":
#     MODEL_NAME = "openai/clip-vit-base-patch32"
#     CLASSES = ["cat", "dog", "bird", "fish"]
#
#     TOKENIZER = CLIPTokenizerFast.from_pretrained(MODEL_NAME)
#
#     print("--- Example 1: Full Fine-Tuning ---")
#     model_to_train, zero_shot_eval_model = set_up_clip(
#         model_name=MODEL_NAME,
#         classes_labels=CLASSES,
#         tokenizer=TOKENIZER,
#         freeze_text=False,
#         freeze_image=False,
#     )
#
#     print("\n--- Example 3: Zero-Shot Baseline Setup ---")
#     _, zero_shot_baseline = set_up_clip(
#         model_name=MODEL_NAME,
#         classes_labels=CLASSES,
#         tokenizer=TOKENIZER,
#         freeze_text=True,
#         freeze_image=True
#     )


# --- Example Usage (How to use the function) ---
if __name__ == "__main__":

    # 1. Define base parameters
    MODEL_NAME = "openai/clip-vit-base-patch32"
    DEVICE = torch.device("mps")
    print(f"Using device: {DEVICE}")

    # 2. Load Tokenizer (ImageProcessor is handled by setup_dataset_realtime's defaults)
    TOKENIZER = CLIPTokenizerFast.from_pretrained(MODEL_NAME)

    # 3. --- Setup STL-10 (Cat/Dog) Dataset ---
    print("\n--- 1. Setting up STL-10 (Cat/Dog) Dataset ---")
    try:
        # Get the base dataset (Subset of cats and dogs)
        base_ds = get_STL_dataset(folder='data/STL10')

        # setup_dataset_realtime will split, apply default (CLIP-compatible) transforms
        # and create DataLoaders
        bundle = setup_dataset_realtime(
            dataset=base_ds,
            batch_size=64,
            num_workers=0  # Use 0 for main process, easier for __main__
        )

        CLASSES = bundle.classes
        test_loader = bundle.test_loader

        print(f"Dataset ready. Classes found: {CLASSES}")  # Should be ['cat', 'dog']

        # 4. --- Setup Zero-Shot CLIP Model ---
        print("\n--- 2. Setting up Zero-Shot CLIP Model ---")

        # We only need the second return value (the classifier) for this example
        _, zero_shot_baseline = set_up_clip(
            model_name=MODEL_NAME,
            classes_labels=CLASSES,
            tokenizer=TOKENIZER,
            freeze_text=True,  # Freeze both for zero-shot
            freeze_image=True
        )

        # 5. --- Evaluate Zero-Shot Model ---
        print("\n--- 3. Evaluating Zero-Shot Model ---")

        # calculate_accuracy will move model to device
        accuracy = calculate_accuracy(zero_shot_baseline, test_loader)

        print(f"\n--- Result ---")
        print(f"Zero-shot accuracy on STL-10 (Cats vs Dogs): {accuracy * 100:.2f}%")
        print("Note: This model has never seen an STL-10 image!")

    except ImportError:
        print("\nSkipping __main__ example: `dataset.py` not found.")
    except Exception as e:
        print(f"\nAn error occurred during the example run: {e}")
        import traceback

        traceback.print_exc()