import numpy as np


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    return xyxy


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Non-Maximum Suppression with xyxy boxes."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return keep


def scale_boxes(boxes: np.ndarray, scale: float, pad: tuple[int, int]) -> np.ndarray:
    """Scale boxes from padded 640x640 space back to original image coordinates."""
    pad_w, pad_h = pad
    boxes[:, 0] -= pad_w
    boxes[:, 1] -= pad_h
    boxes[:, 2] -= pad_w
    boxes[:, 3] -= pad_h
    boxes /= scale
    return boxes


def postprocess(
    output: np.ndarray,
    scales: list[float],
    pads: list[tuple[int, int]],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> list[dict]:
    """
    Postprocess raw YOLO output.

    Args:
        output:          [N, 84, 8400] raw model output
        scales:          scale factor per image
        pads:            (pad_w, pad_h) per image
        conf_threshold:  minimum confidence to keep
        iou_threshold:   NMS IoU threshold

    Returns:
        list of dicts per image, each with:
            boxes:    [M, 4] np.ndarray (x1, y1, x2, y2) in original image coords
            scores:   [M] np.ndarray
            class_ids: [M] np.ndarray
    """
    batch_size = output.shape[0]
    # [N, 84, 8400] -> [N, 8400, 84]
    output = output.transpose(0, 2, 1)

    results = []
    for i in range(batch_size):
        preds = output[i]  # [8400, 84]

        # Split: first 4 = box coords, rest = class scores
        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]

        # Best class per detection
        class_ids = class_scores.argmax(axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]

        # Filter by confidence
        mask = confidences >= conf_threshold
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            results.append({"boxes": np.empty((0, 4)), "scores": np.empty(0), "class_ids": np.empty(0, dtype=int)})
            continue

        # Convert to xyxy
        boxes_xyxy = xywh_to_xyxy(boxes_xywh)

        # NMS
        keep = nms(boxes_xyxy, confidences, iou_threshold)
        boxes_xyxy = boxes_xyxy[keep]
        confidences = confidences[keep]
        class_ids = class_ids[keep]

        # Scale back to original image
        boxes_xyxy = scale_boxes(boxes_xyxy, scales[i], pads[i])

        results.append({
            "boxes": boxes_xyxy,
            "scores": confidences,
            "class_ids": class_ids,
        })

    return results
