from ultralytics import YOLO
from transformers import Sam2Processor, Sam2Model
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def create_color_palette(num_classes):
    """Создает простую, но различимую цветовую палитру."""
    cmap = plt.get_cmap('hsv', num_classes)
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        color_rgb = (np.array(cmap(i)[:3]) * 255).astype(np.uint8)
        palette[i] = color_rgb
    return palette


device = "cpu"
print(f"Using device: {device}")

yolo_model = YOLO("yolo11s.pt")

sam_processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large", use_fast=False)
sam_model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(device)

original_image = Image.open("../data/image_to_detect_and_segment.jpg")

original_image_np = np.array(original_image)
H, W, _ = original_image_np.shape

print("Running YOLOv11 object detection...")
confidence_threshold = 0.02
yolo_results = yolo_model(original_image, conf=confidence_threshold)[0]

bboxes = yolo_results.boxes.xyxy.cpu()
class_ids = yolo_results.boxes.cls.cpu().numpy().astype(int)
num_classes = len(yolo_model.names)
print(f"YOLO detected {len(bboxes)} objects.")

if len(bboxes) == 0:
    print("YOLO found no objects. Exiting.")
    exit()

print("Running SAM2 segmentation on detected boxes...")
box_labels = torch.zeros(len(bboxes), dtype=torch.int)
bboxes_list = bboxes.tolist()

inputs = sam_processor(
    original_image,
    input_boxes=[bboxes_list],
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = sam_model(**inputs)

raw_masks = outputs.pred_masks
original_H = inputs["original_sizes"][0][0].item()
original_W = inputs["original_sizes"][0][1].item()
raw_masks = outputs.pred_masks
B, N, N_masks, H_pred, W_pred = raw_masks.shape
raw_masks_reshaped = raw_masks.view(B * N, N_masks, H_pred, W_pred)
interpolated_masks = F.interpolate(
    raw_masks_reshaped,
    size=(original_H, original_W),
    mode='bilinear',
    align_corners=False
)

masks_tensor_all_variants = interpolated_masks
masks_tensor = masks_tensor_all_variants[:, 0, :, :]


masks_np = (masks_tensor.cpu().numpy() > 0.5)

print("Combining results and visualizing...")
palette = create_color_palette(num_classes)
color_mask_image = np.zeros((H, W, 3), dtype=np.uint8)

for i in range(len(masks_np)):
    mask = masks_np[i]
    class_id = class_ids[i]
    color = palette[class_id]

    color_mask_image[mask] = color

mask_image_pil = Image.fromarray(color_mask_image)
overlayed_image = Image.blend(original_image, mask_image_pil, alpha=0.6)

fig, axs = plt.subplots(1, 3, figsize=(20, 8))
axs[0].imshow(original_image)
axs[0].set_title("Original Image")
axs[0].axis("off")

axs[1].imshow(mask_image_pil)
axs[1].set_title("SAM2.1 Masks (colored by YOLO class)")
axs[1].axis("off")

axs[2].imshow(overlayed_image)
axs[2].set_title("Image with YOLO + SAM2.1 Overlay")
axs[2].axis("off")

plt.tight_layout()
plt.show()

unique_labels = np.unique(class_ids)
detected_labels = [yolo_model.names[label_id] for label_id in unique_labels]
print(f"Detected classes (out of {num_classes} COCO classes):")
print(detected_labels)