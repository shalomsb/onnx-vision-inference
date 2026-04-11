import os
import numpy as np
import cv2


# Distinct colors per category (BGR)
CATEGORY_COLORS = [
    (0, 200, 0),      # green
    (255, 0, 0),      # blue
    (0, 0, 255),      # red
    (255, 255, 0),    # cyan
    (255, 0, 255),    # magenta
    (0, 255, 255),    # yellow
    (128, 0, 255),    # purple
    (0, 128, 255),    # orange
]


def draw_detections(
    image: np.ndarray,
    detections: dict,
    labels: list[str] = None,
) -> np.ndarray:
    """
    Draw bounding boxes and scores on an image.

    Args:
        image:      HxWxC BGR uint8
        detections: dict with "boxes" [N, 4] and "scores" [N]
        labels:     per-detection category names (from map_token_to_category)

    Returns:
        annotated image copy
    """
    img = image.copy()
    boxes = detections["boxes"]
    scores = detections["scores"]

    # Assign colors by unique category
    color_map = {}
    color_idx = 0

    for i, ((x1, y1, x2, y2), score) in enumerate(zip(boxes.astype(int), scores)):
        lbl = labels[i] if labels else "object"

        if lbl not in color_map:
            color_map[lbl] = CATEGORY_COLORS[color_idx % len(CATEGORY_COLORS)]
            color_idx += 1
        color = color_map[lbl]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        text = f"{lbl} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, max(y1 - th - 8, 0)), (x1 + tw, y1), color, -1)
        cv2.putText(img, text, (x1, max(y1 - 4, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img


def save_results(
    images: list[np.ndarray],
    results: list[dict],
    labels: list[list[str]] = None,
    output_dir: str = "/app/groundingdino/output",
):
    """Draw detections and save annotated images."""
    os.makedirs(output_dir, exist_ok=True)
    for i, (img, det) in enumerate(zip(images, results)):
        lbls = labels[i] if labels else None
        drawn = draw_detections(img, det, labels=lbls)
        path = os.path.join(output_dir, f"result_{i}.jpg")
        cv2.imwrite(path, drawn)
        print(f"Saved: {path}")
