import os
import random
import numpy as np
from PIL import Image

import torch
import torch.utils.data
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

BATCH_SIZE = 32
DATA_DIR = 'data/muffin-vs-chihuahua-image-classification'
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f).convert('RGBA')
        return img.convert('RGB')

def get_data_transforms():
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

def load_and_split_datasets(data_dir, train_ratio, data_transforms, generator):
    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'test'),
        data_transforms['test'],
        loader=rgb_loader
    )

    full_train_dataset_train_tf = datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        data_transforms['train'],
        loader=rgb_loader
    )

    full_train_dataset_val_tf = datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        data_transforms['test'],
        loader=rgb_loader
    )

    train_size = int(train_ratio * len(full_train_dataset_train_tf))
    val_size = len(full_train_dataset_train_tf) - train_size

    train_dataset, _ = torch.utils.data.random_split(
        full_train_dataset_train_tf,
        [train_size, val_size],
        generator=generator
    )

    _, val_dataset = torch.utils.data.random_split(
        full_train_dataset_val_tf,
        [train_size, val_size],
        generator=generator
    )

    image_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }

    class_names = test_dataset.classes

    return image_datasets, class_names

def create_dataloaders(image_datasets, batch_size, seed_worker_fn, generator):
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=(x == 'train'),
            num_workers=4,
            worker_init_fn=seed_worker_fn,
            generator=generator,
        )
        for x in ['train', 'val', 'test']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    return dataloaders, dataset_sizes

def print_dataset_info(dataset_sizes, class_names, image_datasets):
    print(f'Dataset sizes: {dataset_sizes}')
    print(f'Class names: {class_names}')

    class_distribution = {}
    for split in ['train', 'val', 'test']:
        split_class_counts = {class_name: 0 for class_name in class_names}

        if split in ['train', 'val']:
            subset = image_datasets[split]
            full_dataset_samples = subset.dataset.samples
            indices = subset.indices
            samples_to_check = [full_dataset_samples[i] for i in indices]
        else:
            samples_to_check = image_datasets[split].samples

        for _, label in samples_to_check:
            class_name = class_names[label]
            split_class_counts[class_name] += 1

        class_distribution[split] = split_class_counts

    for split, dist in class_distribution.items():
        print(f'{split.capitalize()} class distribution: {dist}')

def get_device():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    return device

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def calculate_class_weights(train_dataset, device):
    print("Calculating class weights for training set...")

    try:
        labels = [train_dataset.dataset.samples[i][1] for i in train_dataset.indices]
    except AttributeError:
        print("Could not get 'samples' or 'indices' from train_dataset.")
        print("Weights will not be applied.")
        return None
    except Exception as e:
        print(f"An error occurred during weight calculation: {e}")
        print("Weights will not be applied.")
        return None

    class_counts = torch.bincount(torch.tensor(labels))

    if len(class_counts) == 0:
        print("No labels found in training set. Weights will not be applied.")
        return None

    total_samples = float(sum(class_counts))
    num_classes = float(len(class_counts))

    weights = total_samples / (num_classes * class_counts.float())

    print(f"Found {int(total_samples)} samples.")
    for i, count in enumerate(class_counts):
        print(f"  Class {i}: {count} samples, Weight: {weights[i]:.4f}")

    return weights.to(device)

def load_and_prepare_data():
    torch.manual_seed(RANDOM_SEED)
    g = torch.Generator().manual_seed(RANDOM_SEED)

    data_transforms = get_data_transforms()

    try:
        image_datasets, class_names = load_and_split_datasets(
            DATA_DIR, TRAIN_RATIO, data_transforms, g
        )
    except FileNotFoundError:
        print(f"Error: Data directory not found at '{DATA_DIR}'")
        print("Please download and unpack the dataset to the correct location.")
        return

    dataloaders, dataset_sizes = create_dataloaders(
        image_datasets, BATCH_SIZE, seed_worker, g
    )

    print_dataset_info(dataset_sizes, class_names, image_datasets)

    device = get_device()

    print("\nData loading and preparation complete.")

    fig = plt.figure(figsize=(12, 12))
    plt.suptitle("Image examples from datasets (Train, Val, Test)", fontsize=16)

    splits = ['train', 'val', 'test']
    num_images_to_show = 6

    for i, split in enumerate(splits):
        inputs, classes = next(iter(dataloaders[split]))

        out = torchvision.utils.make_grid(inputs[:num_images_to_show])

        ax = fig.add_subplot(3, 1, i + 1)

        out_np = out.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        out_np = std * out_np + mean
        out_np = np.clip(out_np, 0, 1)

        ax.imshow(out_np)
        ax.axis('off')

        labels_title = [class_names[x] for x in classes[:num_images_to_show]]
        ax.set_title(f'{split.capitalize()} (Examples: {", ".join(labels_title)})', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return dataloaders, dataset_sizes, class_names, device


if __name__ == "__main__":
    load_and_prepare_data()
