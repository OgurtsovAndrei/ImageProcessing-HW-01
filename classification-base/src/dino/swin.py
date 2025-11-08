import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models as models
import matplotlib.pyplot as plt
import os
import sys

from dino.dataset import load_and_prepare_data as dataloader_main
from dino.dataset import calculate_class_weights
from dino.train import train_model, evaluate_best_model, visualize_model, get_params_number

LEARNING_RATE = 1e-5
NUM_EPOCHS = 5

def create_swin_model(device, num_classes=2):
    model_swin = models.swin_t(weights='IMAGENET1K_V1')

    num_ftrs = model_swin.head.in_features
    model_swin.head = nn.Linear(num_ftrs, num_classes)

    model_swin = model_swin.to(device)
    return model_swin

def setup_training(model, lr, num_epochs, class_weights=None):
    trainable_params = [param for param in model.parameters() if param.requires_grad]

    print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = optim.AdamW(trainable_params, lr=lr)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    if class_weights is not None:
        print("Using weighted CrossEntropyLoss.")
    else:
        print("Using standard CrossEntropyLoss.")


    return optimizer, scheduler, criterion

def run_swin_experiment():
    print("--- Step 1: Loading and preparing data ---")
    dataloader_result = dataloader_main()

    if dataloader_result is None:
        print("Data loading failed. Exiting.")
        return

    dataloaders, dataset_sizes, class_names, device = dataloader_result

    print("\n--- Step 2: Creating Swin-T model ---")
    model_swin = create_swin_model(device, len(class_names))

    print("\n--- Step 2.5: Calculating weights for imbalanced classes ---")
    class_weights = calculate_class_weights(dataloaders['train'].dataset, device)

    print("\n--- Step 3: Setting up training (optimizer, scheduler, loss) ---")
    optimizer_swin, exp_lr_scheduler, criterion = setup_training(
        model_swin, LEARNING_RATE, NUM_EPOCHS, class_weights=class_weights
    )

    print("\n--- Step 4: Counting model parameters ---")
    get_params_number(model_swin)

    print("\n--- Step 5: Starting model training ---")
    model_swin = train_model(
        model=model_swin,
        criterion=criterion,
        optimizer=optimizer_swin,
        scheduler=exp_lr_scheduler,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        device=device,
        num_epochs=NUM_EPOCHS
    )

    print("\n--- Step 6: Evaluating best model on test set ---")
    evaluate_best_model(
        model=model_swin,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        device=device
    )

    print("\n--- Step 7: Visualizing results ---")
    visualize_model(
        model=model_swin,
        dataloaders=dataloaders,
        device=device,
        class_names=class_names
    )

    plt.ioff()
    plt.show()

    print("\n--- Process finished ---")

if __name__ == "__main__":
    run_swin_experiment()

