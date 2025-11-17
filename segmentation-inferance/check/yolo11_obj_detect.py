from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import cv2

model = YOLO("yolo11x.pt")

original_image = Image.open("../data/image_to_detect_and_segment.jpg")

results_list = model(original_image, conf=0.05)
result = results_list[0]

print("Detected objects (YOLOv11x, conf > 0.05):")

for box in result.boxes:
    box_coords = [round(i, 2) for i in box.xyxy[0].tolist()]
    label_name = result.names[box.cls[0].item()]
    confidence = round(box.conf[0].item(), 3)

    print(
        f"  Detected {label_name} with confidence "
        f"{confidence} at location {box_coords}"
    )

image_with_boxes_bgr = result.plot()

image_with_boxes_rgb = cv2.cvtColor(image_with_boxes_bgr, cv2.COLOR_BGR2RGB)

fig, axs = plt.subplots(1, 2, figsize=(16, 8))

axs[0].imshow(original_image)
axs[0].set_title("Original Image")
axs[0].axis("off")

axs[1].imshow(image_with_boxes_rgb)
axs[1].set_title("Image with YOLOv11 detections (conf > 0.05)")
axs[1].axis("off")

plt.tight_layout()
plt.show()