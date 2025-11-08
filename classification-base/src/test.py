import os
import re
from typing import Dict, Tuple, Callable, List, Optional

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.swin import load_swin_from_weights
from src.model.vit import load_vit_from_weights
from src.model.resnet import load_resnet_from_weights
from src.data.dataset import setup_dataset_realtime, DatasetBundle, get_STL_dataset

WEIGHTS_DIR = 'models/weights'
DATASET_NAME = 'STL10'
DATA_ROOT = 'data/STL10'  # default dataset root
RESULTS_CSV_PATH = 'test_results.csv'
BATCH_SIZE = 64


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("Using CUDA.")
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using Apple MPS.")
        return torch.device("mps")
    print("Using CPU.")
    return torch.device("cpu")


def find_best_models(weights_dir: str, include_prefixes: Optional[List[str]] = None) -> Dict[str, Dict[str, any]]:
    best_models = {}
    pattern = re.compile(r"^(.*?)-\d{8}_\d{6}-f1_([\d.]+)\.pth$")

    print(f"Searching for best models in {weights_dir} (recursively)...")
    for root, _, files in os.walk(weights_dir):
        for filename in files:
            if not filename.endswith(".pth"):
                continue
            match = pattern.match(filename)
            if not match:
                continue
            model_name, f1_str = match.groups()
            if include_prefixes and not any(model_name.startswith(p) for p in include_prefixes):
                continue
            f1_score_val = float(f1_str)

            full_path = os.path.join(root, filename)
            if model_name not in best_models or f1_score_val > best_models[model_name]['best_val_f1']:
                best_models[model_name] = {
                    'path': full_path,
                    'best_val_f1': f1_score_val
                }

    print(f"Found {len(best_models)} unique models.")
    for name, data in best_models.items():
        print(f"  - Model: {name}, Best Val F1: {data['best_val_f1']:.4f}, Path: {data['path']}")

    return best_models


def get_model_loader(model_name: str) -> Callable[[str, int], nn.Module]:
    loaders = {
        'vit': load_vit_from_weights,
        'swin': load_swin_from_weights,
        'resnet': load_resnet_from_weights,
    }

    for key, loader_func in loaders.items():
        if key in model_name.lower():
            return loader_func

    raise ValueError(f"Unknown model name: {model_name}. "
                     f"Add its loader to the `loaders` dictionary in the `get_model_loader` function.")


@torch.no_grad()
def test_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(dataloader, desc="Testing"):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return test_acc, test_f1


def run_tests(
        bundle: DatasetBundle,
        weights_dir: str = WEIGHTS_DIR,
        results_csv_path: str = RESULTS_CSV_PATH,
        include_prefixes: Optional[List[str]] = None,
):
    device = get_device()

    best_models_info = find_best_models(weights_dir, include_prefixes=include_prefixes)
    if not best_models_info:
        print("No suitable weight files (.pth) found in the models/weights directory.")
        print("Ensure that the files are in the format 'model-name-DATE_TIME-f1_SCORE.pth'")
        return None

    try:
        print("Using provided dataset bundle for testing (preferred path).")
        test_loader = bundle.test_loader
        effective_num_classes = bundle.num_classes
    except Exception as e:
        print(f"Failed to obtain dataset/test loader: {e}")
        return None

    results = []
    for model_name, info in best_models_info.items():
        print(f"\n--- Testing model: {model_name} ---")
        try:
            model_loader = get_model_loader(model_name)

            model = model_loader(weights_path=info['path'], num_classes=effective_num_classes)
            print(f"Model created and weights loaded from: {info['path']}")

            test_acc, test_f1 = test_model(model, test_loader, device)
            print(f"Test results: Accuracy = {test_acc:.4f}, F1-score = {test_f1:.4f}")

            results.append({
                'model_name': model_name,
                'best_val_f1': info['best_val_f1'],
                'test_accuracy': test_acc,
                'test_f1_score': test_f1,
                'weights_path': info['path']
            })
        except Exception as e:
            print(f"Failed to test model {model_name}. Error: {e}")

    if results:
        results_df = pd.DataFrame(results)

        print("\n--- Final Results ---")
        pd.set_option('display.precision', 4)
        pd.set_option('display.max_colwidth', None)
        print(results_df.to_string(index=False))

        results_df.to_csv(results_csv_path, index=False)
        print(f"\nResults saved to file: {results_csv_path}")
        return results_df
    else:
        print("\nNo models were tested.")
        return None


def main():
    run_tests(
        weights_dir=WEIGHTS_DIR,
        results_csv_path=RESULTS_CSV_PATH,
        include_prefixes=None,
    )


if __name__ == '__main__':
    main()
