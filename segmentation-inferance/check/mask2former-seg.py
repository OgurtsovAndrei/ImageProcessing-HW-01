from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt

image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-ade-semantic")

image = Image.open("../data/image_to_detect_and_segment.jpg")
inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

pred_semantic_map = image_processor.post_process_semantic_segmentation(
    outputs,
    target_sizes=[(image.height, image.width)]
)[0]

def create_color_palette(num_classes):
    cmap = plt.cm.get_cmap('hsv', num_classes)
    palette = np.zeros((num_classes, 3), dtype=np.uint8)

    for i in range(num_classes):
        color_rgb = (np.array(cmap(i)[:3]) * 255).astype(np.uint8)
        palette[i] = color_rgb

    return palette


seg_map_numpy = pred_semantic_map.cpu().numpy()

num_classes = model.config.num_labels
palette = create_color_palette(num_classes)

color_seg_map = np.zeros((seg_map_numpy.shape[0], seg_map_numpy.shape[1], 3), dtype=np.uint8)

unique_labels = np.unique(seg_map_numpy)

for label_id in unique_labels:
    color_seg_map[seg_map_numpy == label_id] = palette[label_id]

mask_image_pil = Image.fromarray(color_seg_map)

original_image_rgb = image.convert("RGB")

overlayed_image = Image.blend(original_image_rgb, mask_image_pil, alpha=0.6)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

axs[0].imshow(original_image_rgb)
axs[0].set_title("Original Image (from ADE20k)")
axs[0].axis("off")

axs[1].imshow(mask_image_pil)
axs[1].set_title("ADE20k Semantic Mask")
axs[1].axis("off")

axs[2].imshow(overlayed_image)
axs[2].set_title("Image with ADE20k Overlay")
axs[2].axis("off")

plt.tight_layout()
plt.show()

print(f"Detected classes (out of {num_classes} ADE20k classes):")
id2label = model.config.id2label
detected_labels = [id2label[label_id] for label_id in unique_labels]
print(detected_labels)