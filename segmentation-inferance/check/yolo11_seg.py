from ultralytics import YOLO
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def create_color_palette(num_classes):
    cmap = plt.get_cmap('hsv', num_classes)
    palette = np.zeros((num_classes, 3), dtype=np.uint8)

    for i in range(num_classes):
        color_rgb = (np.array(cmap(i)[:3]) * 255).astype(np.uint8)
        palette[i] = color_rgb

    return palette


model = YOLO("yolo11s-seg.pt")

original_image = Image.open("../data/image_to_detect_and_segment.jpg")

confidence_threshold = 0.7
results_list = model(original_image, conf=confidence_threshold)
result = results_list[0]

num_classes = len(result.names)
palette = create_color_palette(num_classes)

original_image_np = np.array(original_image)
H, W, _ = original_image_np.shape
color_mask_image = np.zeros((H, W, 3), dtype=np.uint8)

if result.masks is not None:
    masks_tensor = result.masks.data

    if masks_tensor.shape[1:] != (H, W):
        masks_tensor = F.interpolate(
            masks_tensor.unsqueeze(1),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)

    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    masks_np = (masks_tensor.cpu().numpy() > 0.5)

    print(f"Detected {len(masks_np)} instance masks.")

    for i in range(len(masks_np)):
        mask = masks_np[i]
        class_id = class_ids[i]
        color = palette[class_id]
        color_mask_image[mask] = color
else:
    print("No masks detected.")

mask_image_pil = Image.fromarray(color_mask_image)
overlayed_image = Image.blend(original_image, mask_image_pil, alpha=0.6)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

axs[0].imshow(original_image)
axs[0].set_title("Original Image")
axs[0].axis("off")

axs[1].imshow(mask_image_pil)
axs[1].set_title("YOLOv11 Instance Mask")
axs[1].axis("off")

axs[2].imshow(overlayed_image)
axs[2].set_title("Image with YOLOv11 Overlay")
axs[2].axis("off")

plt.tight_layout()
plt.show()

if result.masks is not None:
    unique_labels = np.unique(class_ids)
    detected_labels = [result.names[label_id] for label_id in unique_labels]
    print(f"Detected classes (out of {num_classes} COCO classes):")
    print(detected_labels)