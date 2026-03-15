import numpy as np
import cv2


def postprocess(
    pred_masks: np.ndarray,
    iou_scores: np.ndarray,
    orig_h: int,
    orig_w: int,
    mask_threshold: float = 0.5,
) -> dict:
    """
    Postprocess SAM2 mask decoder output.

    Args:
        pred_masks:     [N, 1, H, W] float32 (logits)
        iou_scores:     [N, 1] float32
        orig_h:         original image height
        orig_w:         original image width
        mask_threshold: binary threshold (applied after sigmoid)

    Returns:
        dict with:
            masks:  [N, orig_h, orig_w] bool
            scores: [N] float32
    """
    n = pred_masks.shape[0]
    masks = np.zeros((n, orig_h, orig_w), dtype=bool)

    for i in range(n):
        logit = pred_masks[i, 0]  # [H, W]

        # Sigmoid
        prob = 1.0 / (1.0 + np.exp(-logit.clip(-50, 50)))

        # Resize to original image size
        prob_resized = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Binary threshold
        masks[i] = prob_resized >= mask_threshold

    scores = iou_scores[:, 0]  # [N]

    return {"masks": masks, "scores": scores}
