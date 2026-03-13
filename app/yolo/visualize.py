import os
import cv2
import numpy as np

from .labels import COCO_LABELS


# Generate a consistent color per class
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(COCO_LABELS), 3), dtype=np.uint8)


def draw_detections(image: np.ndarray, detections: dict) -> np.ndarray:
    """Draw bounding boxes and labels on an image."""
    img = image.copy()
    boxes = detections["boxes"]
    scores = detections["scores"]
    class_ids = detections["class_ids"]

    for box, score, cls_id in zip(boxes, scores, class_ids):
        cls_id = int(cls_id)
        x1, y1, x2, y2 = box.astype(int)
        color = COLORS[cls_id].tolist()
        label = f"{COCO_LABELS[cls_id]} {score:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def save_results(images: list[np.ndarray], results: list[dict], output_dir: str = "/app/yolo/output"):
    """Draw detections and save images."""
    os.makedirs(output_dir, exist_ok=True)
    for i, (img, det) in enumerate(zip(images, results)):
        drawn = draw_detections(img, det)
        path = os.path.join(output_dir, f"result_{i}.jpg")
        cv2.imwrite(path, drawn)
        print(f"Saved: {path}")
