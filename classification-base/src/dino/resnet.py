import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models as models
import matplotlib.pyplot as plt

from dino.dataset import load_and_prepare_data as data_main
from dino.train import train_model, evaluate_best_model, visualize_model, get_params_number

LEARNING_RATE = 1e-4
NUM_EPOCHS = 5

def create_resnet_model(device, num_classes=2):
    model_resnet = models.resnet152(weights='IMAGENET1K_V1')

    for param in model_resnet.parameters():
        param.requires_grad = False

    num_ftrs = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_ftrs, num_classes)

    model_resnet = model_resnet.to(device)
    return model_resnet

def setup_training(model, lr, num_epochs):
    trainable_params = [param for param in model.parameters() if param.requires_grad]

    print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = optim.AdamW(trainable_params, lr=lr)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    criterion = nn.CrossEntropyLoss()

    return optimizer, scheduler, criterion

def main():
    print("--- Шаг 1: Загрузка и подготовка данных ---")
    dataloaders, dataset_sizes, class_names, device = data_main()

    print("\n--- Шаг 2: Создание модели ResNet ---")
    model_resnet = create_resnet_model(device, len(class_names))

    print("\n--- Шаг 3: Настройка обучения (оптимизатор, планировщик, потери) ---")
    optimizer_resnet, exp_lr_scheduler, criterion = setup_training(
        model_resnet, LEARNING_RATE, NUM_EPOCHS
    )

    print("\n--- Шаг 4: Подсчет параметров модели ---")
    get_params_number(model_resnet)

    print("\n--- Шаг 5: Запуск обучения модели ---")
    model_resnet = train_model(
        model=model_resnet,
        criterion=criterion,
        optimizer=optimizer_resnet,
        scheduler=exp_lr_scheduler,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        device=device,
        num_epochs=NUM_EPOCHS
    )

    print("\n--- Шаг 6: Оценка лучшей модели на тестовом наборе ---")
    evaluate_best_model(model_resnet, dataloaders, dataset_sizes, device)

    print("\n--- Шаг 7: Визуализация результатов ---")
    visualize_model(
        model=model_resnet,
        dataloaders=dataloaders,
        device=device,
        class_names=class_names,
        num_images=6
    )

    plt.ioff()
    plt.show()

    print("\n--- Процесс завершен ---")

if __name__ == "__main__":
    main()

