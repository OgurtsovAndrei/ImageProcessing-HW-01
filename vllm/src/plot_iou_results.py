import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path


def main():
    yolo_file = Path("/Users/andrei.ogurtsov/NUP/ImProc/ImageProcessing-HW-01/vllm/results/yolo_iou_results.txt")
    if not yolo_file.exists():
        print(f"Error: {yolo_file} not found.")
        return
    yolo_data = pd.read_csv(yolo_file, sep="\t")
    yolo_curve = yolo_data[yolo_data['Size'] > 0]
    vlm_file = Path(
        "/Users/andrei.ogurtsov/NUP/ImProc/ImageProcessing-HW-01/vllm/results/qwen2_vl_fine_tune_results.txt")
    if not vlm_file.exists():
        print(f"Error: {vlm_file} not found.")
        return
    vlm_data = pd.read_csv(vlm_file, sep="\t")
    zero_shot_vlm_iou = 0.2038
    gemini_zero_shot_iou = 0.6472
    zero_shot_yolo_iou = 0.2498
    plt.figure(figsize=(10, 6))
    plt.plot(yolo_curve['Size'], yolo_curve['Mean_IoU'], marker='o', label='Specialized Detector (YOLOv8n)')
    plt.plot(vlm_data['Size'], vlm_data['Mean_IoU'], marker='s', label='Fine-tuned VLM (Qwen2-VL-2B)')
    plt.axhline(y=zero_shot_vlm_iou, color='r', linestyle='--', label='Zero-shot VLM (Qwen2-VL-2B)')
    plt.axhline(y=gemini_zero_shot_iou, color='g', linestyle='-.', label='Zero-shot Gemini 3 Flash')
    plt.axhline(y=zero_shot_yolo_iou, color='m', linestyle=':', label='Zero-shot YOLOv8n')
    plt.xscale('log', base=2)
    plt.xticks(yolo_curve['Size'], yolo_curve['Size'])
    plt.xlabel('Number of Training Samples (log scale)')
    plt.ylabel('Mean IoU')
    plt.title('Comparison of Object Localization Performance (Mean IoU)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.ylim(0, 1.0)
    output_path = Path("/Users/andrei.ogurtsov/NUP/ImProc/ImageProcessing-HW-01/vllm/plots/results_iou_plot.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
