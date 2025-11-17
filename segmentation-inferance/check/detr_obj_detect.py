from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw, ImageFont
import requests
import matplotlib.pyplot as plt

original_image = Image.open("../data/image_to_detect_and_segment.jpg")

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101", revision="no_timm")

inputs = processor(images=original_image, return_tensors="pt")
outputs = model(**inputs)

target_sizes = torch.tensor([original_image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.05)[0]

image_with_boxes = original_image.copy()
draw = ImageDraw.Draw(image_with_boxes)

print("Detected objects:")

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box_coords = [round(i, 2) for i in box.tolist()]
    label_name = model.config.id2label[label.item()]
    confidence = round(score.item(), 3)

    print(
        f"  Detected {label_name} with confidence "
        f"{confidence} at location {box_coords}"
    )

    draw.rectangle(box_coords, outline="red", width=3)

    text = f"{label_name}: {confidence}"
    draw.rectangle(
        (box_coords[0], box_coords[1], box_coords[0] + len(text) * 6, box_coords[1] + 12),
        fill="red"
    )
    draw.text((box_coords[0], box_coords[1]), text, fill="white")

fig, axs = plt.subplots(1, 2, figsize=(16, 8))

axs[0].imshow(original_image)
axs[0].set_title("Original Image")
axs[0].axis("off")

axs[1].imshow(image_with_boxes)
axs[1].set_title("Image with DETR detections (threshold > 0.05)")
axs[1].axis("off")

plt.tight_layout()
plt.show()