import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, PeftModel
from qwen_vl_utils import process_vision_info
import shutil
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from zero_shot_detection import parse_model_output, calculate_iou, load_yolo_annotation, evaluate_predictions


def train_one_epoch(model, processor, train_files, train_labels_dir, optimizer, device, prompt):
    model.train()
    total_loss = 0
    np.random.shuffle(train_files)
    for img_path in tqdm(train_files, desc="Training", leave=False):
        try:
            image = Image.open(img_path).convert("RGB")
            w, h = image.size
            boxes = load_yolo_annotation(train_labels_dir / (img_path.stem + ".txt"))
            obj_list = [{"name": "macbook",
                         "bbox": [int((b[0] - b[2] / 2) * w), int((b[1] - b[3] / 2) * h), int((b[0] + b[2] / 2) * w),
                                  int((b[1] + b[3] / 2) * h)]} for b in boxes]
            target_text = json.dumps({"objects": obj_list})
            messages = [
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": target_text}]}
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(device)
            labels = inputs["input_ids"].clone()
            prompt_text = processor.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
            prompt_inputs = processor(text=[prompt_text], images=image_inputs, return_tensors="pt").to(device)
            labels[:, :prompt_inputs["input_ids"].shape[1]] = -100
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            del inputs, labels, outputs, image_inputs
        except Exception as e:
            print(f"Error in training step: {e}")
            continue
    return total_loss / len(train_files) if len(train_files) > 0 else 0


def validate(model, processor, val_files, val_labels_dir, device, prompt):
    model.eval()
    all_gt_boxes = []
    all_pred_boxes = []
    total_loss = 0
    for img_path in tqdm(val_files, desc="Validating", leave=False):
        try:
            image = Image.open(img_path).convert("RGB")
            w, h = image.size
            gt_boxes = load_yolo_annotation(val_labels_dir / (img_path.stem + ".txt"))
            all_gt_boxes.append(gt_boxes)
            obj_list = [{"name": "macbook",
                         "bbox": [int((b[0] - b[2] / 2) * w), int((b[1] - b[3] / 2) * h), int((b[0] + b[2] / 2) * w),
                                  int((b[1] + b[3] / 2) * h)]} for b in gt_boxes]
            target_text = json.dumps({"objects": obj_list})
            messages_train = [
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": target_text}]}
            ]
            text_train = processor.apply_chat_template(messages_train, tokenize=False, add_generation_prompt=False)
            image_inputs, _ = process_vision_info(messages_train)
            inputs_train = processor(text=[text_train], images=image_inputs, return_tensors="pt").to(device)
            labels = inputs_train["input_ids"].clone()
            prompt_text = processor.apply_chat_template(messages_train[:1], tokenize=False, add_generation_prompt=True)
            prompt_inputs = processor(text=[prompt_text], images=image_inputs, return_tensors="pt").to(device)
            labels[:, :prompt_inputs["input_ids"].shape[1]] = -100
            with torch.no_grad():
                outputs = model(**inputs_train, labels=labels)
                total_loss += outputs.loss.item()
                messages_eval = [
                    {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
                text_eval = processor.apply_chat_template(messages_eval, tokenize=False, add_generation_prompt=True)
                inputs_eval = processor(text=[text_eval], images=image_inputs, return_tensors="pt").to(device)
                generated_ids = model.generate(**inputs_eval, max_new_tokens=256, do_sample=False)
                output_text = \
                processor.batch_decode([g[len(i):] for i, g in zip(inputs_eval["input_ids"], generated_ids)],
                                       skip_special_tokens=True)[0]
                all_pred_boxes.append(parse_model_output(output_text, w, h))
            del inputs_train, inputs_eval, labels, image_inputs
        except Exception as e:
            print(f"Error in validation step: {e}")
            all_pred_boxes.append([])
            continue
    metrics = evaluate_predictions(all_gt_boxes, all_pred_boxes)
    metrics['val_loss'] = total_loss / len(val_files) if len(val_files) > 0 else 0
    return metrics


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    data_root = "/Users/andrei.ogurtsov/NUP/ImProc/ImageProcessing-HW-01/vllm/data"
    test_images_dir = Path(data_root) / "test" / "images"
    test_labels_dir = Path(data_root) / "test" / "labels"
    val_images_dir = Path(data_root) / "valid" / "images"
    val_labels_dir = Path(data_root) / "valid" / "labels"
    subset_sizes = [2, 4, 8, 16, 32, 64, 128, 256]
    prompt = """Detect all MacBook laptops (Apple laptops) in this image. 
Please provide the bounding box coordinates in JSON format:
{
  "objects": [
    {"name": "macbook", "bbox": [x_min, y_min, x_max, y_max]}
  ]
}"""
    results = []
    num_epochs = 3
    test_files = sorted(list(test_images_dir.glob("*.jpg")))
    val_files = sorted(list(val_images_dir.glob("*.jpg")))
    results_file_path = "/Users/andrei.ogurtsov/NUP/ImProc/ImageProcessing-HW-01/vllm/results/qwen2_vl_fine_tune_results.txt"
    done_sizes = []
    if os.path.exists(results_file_path):
        with open(results_file_path, "r") as f:
            for line in f:
                if "Size" in line: continue
                parts = line.split()
                if len(parts) >= 1:
                    done_sizes.append(int(parts[0]))
    else:
        with open(results_file_path, "w") as f:
            f.write("Size\tmAP@0.5\tMean_IoU\n")
    for size in subset_sizes:
        if size in done_sizes:
            print(f"Size {size} already done, skipping.")
            continue
        print(f"\n--- Fine-tuning Qwen2-VL with subset size: {size} ---")
        subset_dir = Path(data_root) / f"subset_{size}"
        if not subset_dir.exists():
            print(f"Subset dir {subset_dir} not found, skipping.")
            continue
        output_dir = Path(f"/Users/andrei.ogurtsov/NUP/ImProc/ImageProcessing-HW-01/vllm/src/vlm_training/size_{size}")
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map=None
            ).to(device)
            processor = AutoProcessor.from_pretrained(model_name, min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
            lora_config = LoraConfig(
                r=8, lora_alpha=16,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
            )
            weights_path = output_dir / "lora_weights"
            if weights_path.exists():
                print(f"Loading existing weights for size {size} from {weights_path}")
                model = PeftModel.from_pretrained(model, weights_path)
                history_path = output_dir / "history.csv"
                if history_path.exists():
                    history = pd.read_csv(history_path).to_dict('records')
            else:
                model = get_peft_model(model, lora_config)
                train_images_dir = subset_dir / "images"
                train_labels_dir = subset_dir / "labels"
                train_files = sorted(list(train_images_dir.glob("*.jpg")))
                optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
                history = []
                for epoch in range(num_epochs):
                    train_loss = train_one_epoch(model, processor, train_files, train_labels_dir, optimizer, device,
                                                 prompt)
                    val_metrics = validate(model, processor, val_files[:10], val_labels_dir, device, prompt)
                    print(
                        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['val_loss']:.4f}, Val mAP: {val_metrics['map50']:.4f}")
                    history.append({
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'val_loss': val_metrics['val_loss'],
                        'val_map50': val_metrics['map50'],
                        'val_mean_iou': val_metrics['mean_iou']
                    })
                pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
                model.save_pretrained(output_dir / "lora_weights")
            print(f"Evaluating final model for size {size} on full test set...")
            test_metrics = validate(model, processor, test_files, test_labels_dir, device, prompt)
            print(
                f"Size {size} Test mAP@0.5: {test_metrics['map50']:.4f}, Test Mean IoU: {test_metrics['mean_iou']:.4f}")
            results.append((size, test_metrics['map50'], test_metrics['mean_iou']))
            with open(results_file_path, "a") as f:
                f.write(f"{size}\t{test_metrics['map50']:.4f}\t{test_metrics['mean_iou']:.4f}\n")
            with open(output_dir / "test_metrics.json", "w") as f:
                json.dump(test_metrics, f, indent=2)
        except Exception as e:
            print(f"Error during training for size {size}: {e}")
        finally:
            if 'model' in locals(): del model
            if 'processor' in locals(): del processor
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
    print("\nFinal Results:")
    print("Size\tmAP@0.5\tMean_IoU")
    for size, mAP, iou in results:
        print(f"{size}\t{mAP:.4f}\t{iou:.4f}")


if __name__ == "__main__":
    main()
