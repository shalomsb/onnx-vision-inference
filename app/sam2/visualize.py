import os
import numpy as np
import cv2


# Distinct colors for mask overlays (BGR)
MASK_COLORS = [
    (255, 0, 0),      # blue
    (0, 255, 0),      # green
    (0, 0, 255),      # red
    (255, 255, 0),    # cyan
    (255, 0, 255),    # magenta
    (0, 255, 255),    # yellow
    (128, 0, 255),    # purple
    (0, 128, 255),    # orange
]


def draw_masks(
    image: np.ndarray,
    detections: dict,
    prompts: dict = None,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Draw semi-transparent mask overlays, prompt markers, and IoU scores.

    Args:
        image:      HxWxC BGR uint8
        detections: dict with "masks" [N, H, W] bool and "scores" [N] float
        prompts:    optional dict with "boxes" [M, 4] and/or "points" [K, 3] (x, y, label)
        alpha:      mask overlay transparency

    Returns:
        annotated image copy
    """
    img = image.copy()
    masks = detections["masks"]
    scores = detections["scores"]

    for i, (mask, score) in enumerate(zip(masks, scores)):
        color = MASK_COLORS[i % len(MASK_COLORS)]

        # Semi-transparent overlay
        overlay = img.copy()
        overlay[mask] = color
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Draw mask contour
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, color, 2)

        # IoU score label at mask centroid
        ys, xs = np.where(mask)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            label = f"IoU:{score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (cx - 2, cy - th - 6), (cx + tw + 2, cy + 2), (0, 0, 0), -1)
            cv2.putText(img, label, (cx, cy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw prompt markers if provided
    if prompts:
        if "boxes" in prompts:
            for box in prompts["boxes"]:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if "points" in prompts:
            for pt in prompts["points"]:
                x, y, label = int(pt[0]), int(pt[1]), int(pt[2])
                # Foreground = green circle, background = red circle
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                cv2.circle(img, (x, y), 6, color, -1)
                cv2.circle(img, (x, y), 6, (255, 255, 255), 2)

    return img


def save_results(
    images: list[np.ndarray],
    results: list[dict],
    prompts: list[dict] = None,
    output_dir: str = "/app/sam2/output",
):
    """Draw masks and save annotated images."""
    os.makedirs(output_dir, exist_ok=True)
    for i, (img, det) in enumerate(zip(images, results)):
        p = prompts[i] if prompts else None
        drawn = draw_masks(img, det, prompts=p)
        path = os.path.join(output_dir, f"result_{i}.{i}.jpg")
        cv2.imwrite(path, drawn)
        print(f"Saved: {path}")
