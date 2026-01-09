import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path


def main():
    yolo_file = Path(
        "/Users/andrei.ogurtsov/NUP/ImProc/ImageProcessing-HW-01/vllm/results/yolo_progressive_results.txt")
    yolo_data = pd.read_csv(yolo_file, sep="\t")
    vlm_file = Path(
        "/Users/andrei.ogurtsov/NUP/ImProc/ImageProcessing-HW-01/vllm/results/qwen2_vl_fine_tune_results.txt")
    vlm_data = pd.read_csv(vlm_file, sep="\t")
    zero_shot_vlm_map50 = 0.0269
    gemini_zero_shot_map50 = 0.1130
    zero_shot_yolo_map50 = 0.0438
    plt.figure(figsize=(10, 6))
    plt.plot(yolo_data['Size'], yolo_data['mAP@0.5'], marker='o', label='Specialized Detector (YOLOv8n)')
    plt.plot(vlm_data['Size'], vlm_data['mAP@0.5'], marker='s', label='Fine-tuned VLM (Qwen2-VL-2B)')
    plt.axhline(y=zero_shot_vlm_map50, color='r', linestyle='--', label='Zero-shot VLM (Qwen2-VL-2B)')
    plt.axhline(y=gemini_zero_shot_map50, color='g', linestyle='-.', label='Zero-shot Gemini 3 Flash')
    plt.axhline(y=zero_shot_yolo_map50, color='m', linestyle=':', label='Zero-shot YOLOv8n')
    plt.xscale('log', base=2)
    plt.xticks(yolo_data['Size'], yolo_data['Size'])
    plt.xlabel('Number of Training Samples (log scale)')
    plt.ylabel('mAP@0.5')
    plt.title('Comparison of Object Detection Performance')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    output_path = Path("/Users/andrei.ogurtsov/NUP/ImProc/ImageProcessing-HW-01/vllm/plots/results_plot.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
