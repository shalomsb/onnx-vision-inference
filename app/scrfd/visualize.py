import os
import cv2
import numpy as np


# Keypoint colors (BGR): right_eye, left_eye, nose, right_mouth, left_mouth
KP_COLORS = [
    (0, 0, 255),     # right eye — red
    (255, 0, 0),     # left eye — blue
    (0, 255, 0),     # nose — green
    (255, 255, 0),   # right mouth — cyan
    (255, 0, 255),   # left mouth — magenta
]


def draw_detections(image: np.ndarray, detections: dict) -> np.ndarray:
    """Draw face bounding boxes, confidence labels, and keypoints."""
    img = image.copy()
    boxes = detections["boxes"]
    scores = detections["scores"]
    keypoints = detections["keypoints"]

    for box, score, kps in zip(boxes, scores, keypoints):
        x1, y1, x2, y2 = box.astype(int)
        color = (0, 255, 0)  # green bbox
        label = f"{score:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw 5 facial keypoints
        for k in range(5):
            px, py = int(kps[k, 0]), int(kps[k, 1])
            cv2.circle(img, (px, py), 3, KP_COLORS[k], -1)

    return img


def save_results(images: list[np.ndarray], results: list[dict], output_dir: str = "/app/scrfd/output"):
    """Draw detections and save images."""
    os.makedirs(output_dir, exist_ok=True)
    for i, (img, det) in enumerate(zip(images, results)):
        drawn = draw_detections(img, det)
        path = os.path.join(output_dir, f"result_{i}.jpg")
        cv2.imwrite(path, drawn)
        print(f"Saved: {path}")
