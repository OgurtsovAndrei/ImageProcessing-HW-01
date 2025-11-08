import os
from datetime import datetime
from typing import List

# Torch / Lightning bits are used indirectly via train.py
import torch
from torchvision import datasets

import src.train as train_module
from src.data.dataset import setup_dataset_realtime, get_STL_dataset
from src.model.freeze_utils import FreezeStrategy
from src.model.resnet import create_resnet_classifier
from src.model.swin import create_swin_classifier
from src.model.vit import create_vit_classifier
from src.test import run_tests


def get_model_factory(name: str):
    key = name.lower()
    if key == 'resnet':
        return create_resnet_classifier
    if key == 'swin':
        return create_swin_classifier
    if key == 'vit':
        return create_vit_classifier
    raise ValueError(f"Unsupported model name '{name}'. Expected one of: vit, swin, resnet.")


def prepare_dataset(dataset_name: str, data_root: str, batch_size: int):
    if dataset_name.upper() == 'STL10':
        ds = get_STL_dataset(data_root)
    else:
        ds = datasets.ImageFolder(root=data_root)
    bundle = setup_dataset_realtime(dataset=ds, batch_size=batch_size)
    return bundle

STL_DATA_ROOT = 'data/STL10'
WEIGHTS_DIR = 'models/weights'
NUM_CLASSES = 2
BATCH_SIZE = 32
NUM_EPOCHS = 10
L1_LAMBDA = 1e-6
L2_LAMBDA = 1e-5
LR = 1e-4
FREEZE = FreezeStrategy.PCT70
MODELS = {
    'resnet': create_resnet_classifier,
    'swin': create_swin_classifier,
    'vit': create_vit_classifier,
}
FREEZE_STRATEGIES = [FREEZE.PCT70]


def main():
    # Create run-scoped output directory: models/run-YYYYMMDD_HHMMSS
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('models', f'run-{run_timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory created: {run_dir}")

    # Prepare dataset once (STL10 by default)
    dataset_name = 'STL10'
    bundle = prepare_dataset(dataset_name, STL_DATA_ROOT, BATCH_SIZE)
    effective_num_classes = bundle.num_classes

    device = train_module.get_device()
    print(f"Selected device: {device}")

    # Propagate hyperparameters to training module
    train_module.LR = LR
    train_module.L1_LAMBDA = L1_LAMBDA
    train_module.L2_LAMBDA = L2_LAMBDA

    trained_model_prefixes: List[str] = []

    for name, factory in MODELS.items():
        # Train each FREEZE variant into its own subfolder inside the run directory
        for FREEZE in FREEZE_STRATEGIES:
            model = factory(num_classes=effective_num_classes, freeze=FREEZE)
            model = model.to(device)

            freeze_str = FREEZE.name.lower()
            model_name = f"{name.lower()}-{freeze_str}"
            trained_model_prefixes.append(model_name)

            # Variant-specific directory: models/run-<ts>/<model-name-with-freeze>/
            model_variant_dir = os.path.join(run_dir, model_name)
            os.makedirs(model_variant_dir, exist_ok=True)

            print(f"\n==== Training {model_name} for {NUM_EPOCHS} epochs ====")
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=L2_LAMBDA)

            train_module.train_model(
                model=model,
                model_name=model_name,
                train_loader=bundle.train_loader,
                val_loader=bundle.val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                num_epochs=NUM_EPOCHS,
                weights_dir=model_variant_dir,
                num_classes=effective_num_classes,
                class_weights=bundle.class_weights,
            )

    # After training all models, run tests and save one CSV for all
    print("\n==== Running tests for trained models ====")
    run_tests(
        weights_dir=run_dir,
        results_csv_path=os.path.join(run_dir, 'test_results.csv'),
        include_prefixes=trained_model_prefixes,
        bundle=bundle,
    )


if __name__ == '__main__':
    main()
