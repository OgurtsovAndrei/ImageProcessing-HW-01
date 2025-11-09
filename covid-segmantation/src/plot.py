import os
from typing import Mapping, Sequence

import pandas as pd
from matplotlib import pyplot as plt


def plot_loss(history: Mapping[str, Sequence[float]]) -> None:
    plt.plot(history['val_loss'], label='val', marker='o')
    plt.plot(history['train_loss'], label='train', marker='o')
    plt.title('Loss per epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()


def plot_score(history: Mapping[str, Sequence[float]]) -> None:
    plt.plot(history['train_miou'], label='train_mIoU', marker='*')
    plt.plot(history['val_miou'], label='val_mIoU', marker='*')
    plt.title('Score per epoch')
    plt.ylabel('mean IoU')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()


def plot_acc(history: Mapping[str, Sequence[float]]) -> None:
    plt.plot(history['train_acc'], label='train_accuracy', marker='*')
    plt.plot(history['val_acc'], label='val_accuracy', marker='*')
    plt.title('Accuracy per epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()

def generate_plots_from_logs(log_dir: str):
    print(f"Attempting to generate plots from logs in: {log_dir}")
    try:
        metrics_path = os.path.join(log_dir, "metrics.csv")
        if not os.path.exists(metrics_path):
            print(f"Error: Could not find metrics.csv at {metrics_path}")
            return

        metrics_df = pd.read_csv(metrics_path)

        # epoch_metrics = metrics_df.groupby('epoch').mean()
        #
        # history = {
        #     'val_loss': epoch_metrics['val_loss'].dropna().tolist(),
        #     'train_loss': epoch_metrics['train_loss'].dropna().tolist(),
        #     'val_miou': epoch_metrics['val_miou'].dropna().tolist(),
        #     'train_miou': epoch_metrics['train_miou'].dropna().tolist(),
        #     'val_acc': epoch_metrics['val_acc'].dropna().tolist(),
        #     'train_acc': epoch_metrics['train_acc'].dropna().tolist()
        # }

        history = {
            'val_loss': metrics_df['val_loss'].dropna().tolist(),
            'train_loss': metrics_df['train_loss'].dropna().tolist(),
            'val_miou': metrics_df['val_miou'].dropna().tolist(),
            'train_miou': metrics_df['train_miou'].dropna().tolist(),
            'val_acc': metrics_df['val_acc'].dropna().tolist(),
            'train_acc': metrics_df['train_acc'].dropna().tolist()
        }

        print("Generating loss plot...")
        plot_loss(history)
        print("Generating mIoU score plot...")
        plot_score(history)
        print("Generating accuracy plot...")
        plot_acc(history)
        print("Plots generated successfully.")

    except Exception as e:
        print(f"Error generating plots: {e}")
        print("Please check if 'val_loss', 'train_loss', 'val_miou', etc. exist in your metrics.csv")