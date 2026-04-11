import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2] (values stay in same scale)."""
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Pure-NumPy NMS. Returns kept indices sorted by descending score."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    kept = []

    while order.size:
        i = order[0]
        kept.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_threshold]

    return np.array(kept, dtype=np.int64)


def postprocess(
    pred_logits: np.ndarray,
    pred_boxes: np.ndarray,
    orig_size: tuple[int, int],
    score_thresh: float = 0.30,
    iou_thresh: float = 0.50,
    top_k: int = 300,
) -> dict:
    """
    Postprocess Grounding DINO outputs with per-category NMS.

    Args:
        pred_logits:  [1, 900, num_tokens] or [1, 900] raw logits
        pred_boxes:   [1, 900, 4] cx cy w h normalized 0-1
        orig_size:    (orig_h, orig_w)
        score_thresh: minimum score after sigmoid
        iou_thresh:   NMS IoU threshold
        top_k:        max detections before NMS

    Returns:
        dict with:
            boxes:     [N, 4] float32 (x1, y1, x2, y2) in original pixel coords
            scores:    [N] float32
            token_ids: [N] int — which text token each detection matched
    """
    logits = pred_logits[0]  # [900, num_tokens] or [900]
    boxes = pred_boxes[0]    # [900, 4]

    if logits.ndim == 2:
        probs = sigmoid(logits)             # [900, num_tokens]
        scores = probs.max(axis=1)          # [900]
        token_ids = probs.argmax(axis=1)    # [900] — which token each query matches
    else:
        scores = sigmoid(logits)            # [900]
        token_ids = np.zeros(len(scores), dtype=np.int64)

    # Threshold
    keep = scores >= score_thresh
    scores = scores[keep]
    boxes = boxes[keep]
    token_ids = token_ids[keep]

    if len(scores) == 0:
        return {"boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.array([], dtype=np.float32),
                "token_ids": np.array([], dtype=np.int64)}

    # Top-k before NMS
    if len(scores) > top_k:
        idx = np.argsort(scores)[::-1][:top_k]
        scores, boxes, token_ids = scores[idx], boxes[idx], token_ids[idx]

    # cxcywh -> xyxy (normalized)
    boxes_xyxy = cxcywh_to_xyxy(boxes)

    # Per-category NMS: only suppress within the same token group
    all_kept = []
    for tid in np.unique(token_ids):
        mask = token_ids == tid
        indices = np.where(mask)[0]
        kept = nms(boxes_xyxy[mask], scores[mask], iou_threshold=iou_thresh)
        all_kept.append(indices[kept])

    all_kept = np.concatenate(all_kept) if all_kept else np.array([], dtype=np.int64)

    boxes_xyxy = boxes_xyxy[all_kept]
    scores = scores[all_kept]
    token_ids = token_ids[all_kept]

    # Normalized -> absolute pixel coords
    orig_h, orig_w = orig_size
    scale = np.array([orig_w, orig_h, orig_w, orig_h], dtype=np.float32)
    boxes_px = np.clip(boxes_xyxy, 0.0, 1.0) * scale

    return {"boxes": boxes_px.astype(np.float32),
            "scores": scores.astype(np.float32),
            "token_ids": token_ids}
