import time
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

CHECKPOINT_DIR = 'models/checkpoints'

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25):

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_model_params_path = ""

    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train' and scheduler:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc

                timestamp = int(time.time())
                model_name = model.__class__.__name__
                acc_str = f"{epoch_acc:.4f}".replace('.', '_')
                new_filename = f"{model_name}_{timestamp}_{acc_str}.pth"
                new_best_path = os.path.join(CHECKPOINT_DIR, new_filename)

                # Сохраняем новую лучшую модель (предыдущие не удаляем)
                torch.save(model.state_dict(), new_best_path)

                # Обновляем путь к ЛУЧШЕЙ модели (для загрузки в конце)
                best_model_params_path = new_best_path
                print(f"New best model saved to: {new_best_path}")

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Acc: {best_acc:4f}')

    if best_model_params_path and os.path.exists(best_model_params_path):
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
        print(f"Loaded best model weights from: {best_model_params_path}")
    else:
        print("Warning: No best model checkpoint found to load.")
    return model

def evaluate_best_model(model, dataloaders, dataset_sizes, device):
    model.eval()

    running_corrects = 0

    print("-" * 10)
    print("Starting Final Test Evaluation...")

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / dataset_sizes['test']
    print(f'Final Test Acc with Best Model: {test_acc:.4f}')
    print("-" * 10)

    return test_acc

def visualize_model(model, dataloaders, device, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(10, 7))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')

                true_label = class_names[labels[j]]
                pred_label = class_names[preds[j]]

                ax.set_title(f'True: {true_label}\nPredicted: {pred_label}',
                             color=("green" if true_label == pred_label else "red"))
                # imshow_fn(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def get_params_number(model):
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")
