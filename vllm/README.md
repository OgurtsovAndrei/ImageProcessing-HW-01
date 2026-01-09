### Object Detection Task Report (Balanced Test Set Update)

#### Step 1: Dataset Creation and Splitting
- **Dataset**: Custom dataset of MacBook laptops (Apple laptops).
- **Total Images**: 1860.
- **Annotated Images**: 407.
- **Split (Balanced)**: 
  - **Train**: 307 annotated images + ~1353 background images.
  - **Validation**: 50 annotated images + 50 background images.
  - **Test**: 50 annotated images + 50 background images.
- **Note**: The inclusion of 50% background images in the test/validation sets ensures that models are penalized for "hallucinations" (predicting MacBooks where there are none or other laptops).

#### Step 2: Zero-Shot Detection (VLMs)
| Model | mAP@0.5 | Mean IoU |
|-------|---------|----------|
| **Gemini 3 Flash** | **0.7999** | **0.6472** |
| Qwen2-VL-2B | 0.2826 | 0.2038 |

**Observations**: On a balanced test set, Gemini 3 Flash significantly outperforms other zero-shot models. It demonstrates excellent semantic specificity, correctly distinguishing MacBooks from background noise and potentially other laptop brands.

#### Step 3: Zero-Shot Specialized Detector (YOLOv8n)
- **Model**: YOLOv8n (pretrained on COCO, evaluating "laptop" class).
- **Results**:
  - **mAP@0.5**: 0.4385
  - **Mean IoU**: 0.2498
- **Observations**: While YOLOv8n is excellent at finding *laptops* (mAP 0.72 on a MacBook-only test set), its performance drops significantly on a balanced test set. This is because it lacks the specificity to distinguish MacBooks from other laptops, leading to many False Positives on images containing non-MacBook laptops or background objects.

#### Step 4: YOLO Progressive Training
| Training Samples | mAP@0.5 | Mean IoU |
|------------------|---------|----------|
| 0 (Zero-Shot)    | 0.4385  | 0.2498   |
| 2                | 0.2877  | 0.0055   |
| 4                | 0.3218  | 0.0061   |
| 8                | 0.3665  | 0.0057   |
| 16               | 0.3778  | 0.0059   |
| 32               | 0.4892  | 0.0056   |
| 64               | 0.4067  | 0.0058   |
| 128              | 0.5121  | 0.0193   |
| 256              | **0.6311** | 0.0510   |

**Observations**: YOLO performance initially drops (catastrophic forgetting of the general "laptop" features) before recovering as it learns the specific "MacBook" features from the training data.

#### Step 5: VLM Fine-Tuning (Qwen2-VL)
| Training Samples | mAP@0.5 | Mean IoU |
|------------------|---------|----------|
| 0 (Zero-Shot)    | 0.2826  | 0.2038   |
| 2                | 0.2142  | 0.1431   |
| 4                | 0.2234  | 0.1912   |
| 8                | 0.2252  | 0.2040   |
| 16               | 0.3073  | 0.2779   |
| 32               | 0.2400  | 0.2041   |
| 64               | 0.2713  | 0.2343   |
| 128              | 0.1902  | 0.1527   |
| 256              | **0.3142** | 0.2860   |

#### Summary & Comparison
On a balanced test set that strictly punishes false positives, the ranking of models changes significantly:

| Model | Setup | mAP@0.5 | Mean IoU |
|-------|-------|---------|----------|
| **Gemini 3 Flash** | **Zero-Shot** | **0.7999** | **0.6472** |
| YOLOv8n | Fine-Tuned (256) | 0.6311 | 0.0510 |
| YOLOv8n | Zero-Shot | 0.4385 | 0.2498 |
| Qwen2-VL | Fine-Tuned (256) | 0.3142 | 0.2860 |
| Qwen2-VL | Zero-Shot | 0.2826 | 0.2038 |

**Final Conclusions**:
1. **Semantic Specificity Matters**: Large VLMs like Gemini 3 Flash possess superior semantic understanding, allowing them to excel in zero-shot tasks requiring fine-grained distinction (MacBook vs. other laptops).
2. **YOLO's High Recall vs. Precision**: YOLO (especially at low confidence thresholds) is very aggressive at finding anything "laptop-like," which leads to high recall but low precision/IoU on background-heavy datasets.
3. **The Importance of Balanced Evaluation**: Previous evaluations on a positive-only test set gave a misleadingly high score to YOLO's generalist "laptop" detector. Including background images revealed the true performance and the superiority of the large VLM for this specific zero-shot task.

![mAP@0.5 Comparison](plots/results_plot.png)
![Mean IoU Comparison](plots/results_iou_plot.png)
