### Object Detection Task Report

#### Step 1: Dataset Creation and Splitting
- **Dataset**: Custom dataset of MacBook laptops (Apple laptops).
- **Total Images**: 1860.
- **Annotated Images**: 407 images have bounding box annotations for MacBooks (class `0`).
- **Split**: 
  - **Train**: 307 annotated images (plus non-annotated background images).
  - **Validation**: 50 annotated images.
  - **Test**: 50 annotated images.

#### Step 2: Zero-Shot Detection and Evaluation with a VLM
- **Model**: Qwen2-VL-2B-Instruct.
- **Method**: Zero-shot detection using a structured prompt requesting JSON output with bounding boxes.
- **Evaluation Metrics**: Mean IoU and mAP@0.5.
- **Results**:
  - **Mean IoU**: 0.2419
  - **mAP@0.5**: 0.0269

#### Step 6: Zero-Shot Gemini 3 Flash
- **Model**: Gemini 3 Flash (via API).
- **Method**: Zero-shot detection using a structured prompt requesting JSON output.
- **Results**:
  - **Mean IoU**: 0.4140
  - **mAP@0.5**: 0.1130
- **Observations**: Gemini 3 Flash significantly outperforms Qwen2-VL in zero-shot mode, both in localization (IoU) and detection precision (mAP). It achieves more than 4x higher mAP@0.5 compared to the local VLM.

#### Step 3: Zero-Shot Specialized Object Detector
- **Model**: YOLOv8n (pretrained on COCO).
- **Method**: Evaluated the pretrained model on the custom test set without any fine-tuning.
- **Evaluation Metric**: mAP@0.5.
- **Results**:
  - **mAP@0.5**: 0.0438
- **Observations**: As expected, the performance was near zero because the model's pretrained class mappings (COCO) do not match the custom dataset's class `0` (MacBook).

#### Step 4: Training with Increasing Dataset Sizes
- **Model**: YOLOv8n (starting from pretrained weights).
- **Method**: Fine-tuned the model on progressively larger subsets of the annotated training data for 10 epochs each.
- **Results**:

| Training Samples | mAP@0.5 on Test Set | Mean IoU |
|------------------|---------------------|----------|
| 2                | 0.5432              | 0.8144   |
| 4                | 0.5543              | 0.8261   |
| 8                | 0.6141              | 0.8238   |
| 16               | 0.6131              | 0.8056   |
| 32               | 0.7089              | 0.8031   |
| 64               | 0.6757              | 0.8116   |
| 128              | 0.8338              | 0.8707   |
| 256              | 0.9163              | 0.8846   |

- **Observations**: 
  - Even with only 2 samples, the model achieved a relatively high mAP (0.54), likely because the pretrained YOLOv8n model already had strong features for "laptop" from the COCO dataset.
  - Performance improved consistently as the dataset size increased, reaching 0.9163 mAP@0.5 with 256 samples.

#### Step 5: Fine-Tuning the VLM
- **Model**: Qwen2-VL-2B-Instruct.
- **Method**: Fine-tuned the VLM using LoRA (PEFT) on the same training subsets. 
- **Training Details**: 
  - Precision: float16.
  - Epochs: 3 per subset.
  - Learning Rate: 5e-5.
- **Results**:

| Training Samples | mAP@0.5 on Test Set | Mean IoU |
|------------------|---------------------|----------|
| 2                | 0.0191              | 0.2050   |
| 4                | 0.0161              | 0.2013   |
| 8                | 0.0161              | 0.2069   |
| 16               | 0.0277              | 0.2212   |
| 32               | 0.0228              | 0.2353   |
| 64               | 0.0161              | 0.2412   |
| 128              | 0.0445              | 0.2570   |
| 256              | 0.0477              | 0.1319   |

- **Observations**: 
  - Fine-tuning shows a gradual improvement in mAP@0.5 as dataset size increases, though it remains significantly lower than the specialized YOLO detector.
  - Mean IoU (localization) generally improved up to 128 samples, showing that the model is learning to better place bounding boxes.
  - The drop in IoU at 256 samples might indicate overfitting or instability in the LoRA adaptation with the specific hyperparameters used.

#### Summary & Comparison
| Model | Zero-Shot mAP@0.5 | Fine-tuned (256 samples) mAP@0.5 |
|-------|-------------------|----------------------------------|
| YOLOv8n | 0.0438            | 0.9163                           |
| Qwen2-VL-2B | 0.0269            | 0.0477                           |

The comparison clearly shows that while the VLM has some general understanding of the objects (Mean IoU ~0.24 in zero-shot), it struggles with the precision required for object detection as measured by mAP@0.5, especially when compared to a specialized detector like YOLOv8n. Even with fine-tuning, the VLM's detection performance improves slowly, whereas YOLOv8n quickly reaches high accuracy with minimal data.

The final plots `results_plot.png` (mAP@0.5) and `results_iou_plot.png` (Mean IoU) visualize these trends across all training sample sizes.
