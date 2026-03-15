import numpy as np


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


def decode_stride(
    scores: np.ndarray,
    bboxes: np.ndarray,
    kps: np.ndarray,
    stride: int,
    input_size: int,
    conf_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode detections for a single stride level.

    Args:
        scores:         [1, num_anchors, 1] confidence scores
        bboxes:         [1, num_anchors, 4] bbox deltas (distance from anchor)
        kps:            [1, num_anchors, 10] keypoint offsets (5 landmarks x 2)
        stride:         stride value (8, 16, or 32)
        input_size:     model input size (e.g. 640)
        conf_threshold: minimum confidence to keep

    Returns:
        det_boxes:  [N, 4] xyxy
        det_scores: [N]
        det_kps:    [N, 5, 2]
    """
    scores = scores[0]    # [num_anchors, 1]
    bboxes = bboxes[0]    # [num_anchors, 4]
    kps = kps[0]          # [num_anchors, 10]

    feat_h = input_size // stride
    feat_w = input_size // stride

    # Generate anchor centers: grid of (x, y) positions scaled by stride
    # mgrid returns [row_indices, col_indices], [::-1] gives [col (x), row (y)]
    anchor_centers = np.mgrid[:feat_h, :feat_w][::-1].reshape(2, -1).T * stride
    # 2 anchors per position
    anchor_centers = np.repeat(anchor_centers, 2, axis=0)

    # Filter by confidence
    scores_flat = scores[:, 0]
    mask = scores_flat >= conf_threshold
    if not np.any(mask):
        return np.empty((0, 4)), np.empty(0), np.empty((0, 5, 2))

    anchor_centers = anchor_centers[mask]
    scores_flat = scores_flat[mask]
    bboxes = bboxes[mask]
    kps = kps[mask]

    cx = anchor_centers[:, 0]
    cy = anchor_centers[:, 1]

    # Decode boxes: anchor_center -/+ delta * stride -> xyxy
    det_boxes = np.empty((len(cx), 4), dtype=np.float32)
    det_boxes[:, 0] = cx - bboxes[:, 0] * stride  # x1
    det_boxes[:, 1] = cy - bboxes[:, 1] * stride  # y1
    det_boxes[:, 2] = cx + bboxes[:, 2] * stride  # x2
    det_boxes[:, 3] = cy + bboxes[:, 3] * stride  # y2

    # Decode keypoints: anchor_center + offset * stride
    det_kps = np.empty((len(cx), 5, 2), dtype=np.float32)
    for k in range(5):
        det_kps[:, k, 0] = cx + kps[:, k * 2] * stride
        det_kps[:, k, 1] = cy + kps[:, k * 2 + 1] * stride

    return det_boxes, scores_flat, det_kps


def postprocess(
    outputs: list[np.ndarray],
    input_size: int,
    scale: float,
    pad: tuple[int, int],
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.4,
) -> dict:
    """
    Postprocess raw SCRFD outputs.

    SCRFD 10G outputs 9 tensors ordered as:
        [scores_8, scores_16, scores_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32]

    Args:
        outputs:        list of 9 numpy arrays from model
        input_size:     model input size (e.g. 640)
        scale:          letterbox scale factor
        pad:            (pad_w, pad_h) letterbox padding
        conf_threshold: minimum confidence
        iou_threshold:  NMS IoU threshold

    Returns:
        dict with:
            boxes:     [M, 4] np.ndarray (x1, y1, x2, y2) in original image coords
            scores:    [M] np.ndarray
            keypoints: [M, 5, 2] np.ndarray in original image coords
    """
    strides = [8, 16, 32]
    # outputs layout: scores[0:3], bboxes[3:6], kps[6:9]
    all_boxes = []
    all_scores = []
    all_kps = []

    for i, stride in enumerate(strides):
        boxes, scores, keypoints = decode_stride(
            outputs[i],        # scores for this stride
            outputs[i + 3],    # bboxes for this stride
            outputs[i + 6],    # keypoints for this stride
            stride,
            input_size,
            conf_threshold,
        )
        if len(boxes) > 0:
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_kps.append(keypoints)

    if not all_boxes:
        return {
            "boxes": np.empty((0, 4)),
            "scores": np.empty(0),
            "keypoints": np.empty((0, 5, 2)),
        }

    all_boxes = np.concatenate(all_boxes)
    all_scores = np.concatenate(all_scores)
    all_kps = np.concatenate(all_kps)

    # NMS
    keep = nms(all_boxes, all_scores, iou_threshold)
    all_boxes = all_boxes[keep]
    all_scores = all_scores[keep]
    all_kps = all_kps[keep]

    # Scale back to original image coordinates
    pad_w, pad_h = pad
    all_boxes[:, 0] -= pad_w
    all_boxes[:, 1] -= pad_h
    all_boxes[:, 2] -= pad_w
    all_boxes[:, 3] -= pad_h
    all_boxes /= scale

    all_kps[:, :, 0] -= pad_w
    all_kps[:, :, 1] -= pad_h
    all_kps /= scale

    return {
        "boxes": all_boxes,
        "scores": all_scores,
        "keypoints": all_kps,
    }
