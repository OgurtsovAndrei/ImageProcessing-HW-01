import os
from typing import List

import pandas as pd
from matplotlib import pyplot as plt


def save_and_plot_history(history: List[dict], model_name: str, run_timestamp: str, weights_dir: str):
    history_df = pd.DataFrame(history)
    csv_filename = f"{model_name}-{run_timestamp}-metrics.csv"
    csv_path = os.path.join(weights_dir, csv_filename)
    history_df.to_csv(csv_path, index=False)
    print(f"Metrics history saved to {csv_path}")

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history_df['epoch'], history_df['train_loss'], label='Train Loss')
    plt.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{model_name} - Loss vs. Epoch')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history_df['epoch'], history_df['train_acc'], label='Train Accuracy')
    plt.plot(history_df['epoch'], history_df['val_acc'], label='Validation Accuracy')
    plt.plot(history_df['epoch'], history_df['val_f1'], label='Validation F1-score')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.title(f'{model_name} - Metrics vs. Epoch')
    plt.grid(True)

    plt.suptitle(f'Training Metrics for {model_name} ({run_timestamp})')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_filename = f"{model_name}-{run_timestamp}-metrics.png"
    plot_path = os.path.join(weights_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Metrics plot saved to {plot_path}")
