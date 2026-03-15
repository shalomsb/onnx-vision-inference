"""YOLO → SAM2 pipeline: detect objects with YOLO, segment each with SAM2."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time

import numpy as np
import cv2

from yolo.dataloader import create_batch
from yolo.inference import load_model as load_yolo, run_inference as run_yolo
from yolo.postprocess import postprocess as yolo_postprocess

from sam2.preprocess import preprocess as sam2_preprocess, transform_boxes
from sam2.inference import load_models as load_sam2, run_encoder, run_decoder
from sam2.postprocess import postprocess as sam2_postprocess
from sam2.visualize import save_results


# COCO class names (80 classes)
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def main(image_path: str,
         yolo_model: str = "yolo/onnx/yolo11n.onnx",
         sam2_model_dir: str = "sam2/onnx/small",
         gpu: bool = False,
         conf_threshold: float = 0.45,
         iou_threshold: float = 0.45,
         mask_threshold: float = 0.5,
         class_filter: list[int] = None):

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    orig_h, orig_w = image.shape[:2]
    print(f"Image: {image_path} ({orig_w}x{orig_h})")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
    print(f"Using ONNXRuntime providers: {providers}")

    # ── YOLO detection ────────────────────────────────────────────────
    print("\n--- YOLO Detection ---")
    images = [image]
    batch, scales, pads = create_batch(images)

    yolo_session = load_yolo(yolo_model, providers=providers)

    start = time.perf_counter()
    yolo_output = run_yolo(yolo_session, batch)
    yolo_ms = (time.perf_counter() - start) * 1000
    print(f"YOLO inference: {yolo_ms:.2f} ms")

    detections = yolo_postprocess(yolo_output, scales, pads,
                                  conf_threshold=conf_threshold,
                                  iou_threshold=iou_threshold)[0]

    boxes = detections["boxes"]
    scores = detections["scores"]
    class_ids = detections["class_ids"]

    # Filter by class if requested
    if class_filter is not None and len(boxes) > 0:
        mask = np.isin(class_ids, class_filter)
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

    num_det = len(boxes)
    print(f"{num_det} detection(s)")
    for i, (box, score, cls_id) in enumerate(zip(boxes, scores, class_ids)):
        name = COCO_NAMES[int(cls_id)] if int(cls_id) < len(COCO_NAMES) else str(int(cls_id))
        print(f"  [{i}] {name} conf={score:.2f} box={box.astype(int).tolist()}")

    if num_det == 0:
        print("No detections — skipping SAM2.")
        return

    # ── SAM2 segmentation ─────────────────────────────────────────────
    print("\n--- SAM2 Segmentation ---")
    blob, scale_x, scale_y = sam2_preprocess(image)

    sam2_sessions = load_sam2(sam2_model_dir, providers=providers)

    start = time.perf_counter()
    features = run_encoder(sam2_sessions["encoder"], blob)
    encode_ms = (time.perf_counter() - start) * 1000
    print(f"SAM2 encoder: {encode_ms:.2f} ms")

    # Convert YOLO boxes to SAM2 prompts
    point_coords, point_labels = transform_boxes(boxes, scale_x, scale_y)

    # Expand image_embed for batch of prompts
    image_embed = np.tile(features["pix_feat"], (num_det, 1, 1, 1))

    start = time.perf_counter()
    pred_masks, iou_scores = run_decoder(
        sam2_sessions["decoder"],
        point_coords,
        point_labels,
        image_embed,
        features["high_res_feat0"],
        features["high_res_feat1"],
    )
    decode_ms = (time.perf_counter() - start) * 1000
    print(f"SAM2 decoder: {decode_ms:.2f} ms ({num_det} masks)")

    result = sam2_postprocess(pred_masks, iou_scores, orig_h, orig_w,
                              mask_threshold=mask_threshold)

    for i, (score, cls_id) in enumerate(zip(result["scores"], class_ids)):
        name = COCO_NAMES[int(cls_id)] if int(cls_id) < len(COCO_NAMES) else str(int(cls_id))
        area = result["masks"][i].sum()
        print(f"  mask {i}: {name} IoU={score:.3f} area={area} px")

    # Visualize with YOLO boxes overlaid
    prompts = {"boxes": boxes}
    save_results([image], [result], prompts=[prompts],
                 output_dir="/app/sam2/output")


def parse_class_filter(s: str) -> list[int]:
    """Parse comma-separated class IDs: '0,2,5'."""
    return [int(x) for x in s.split(",")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO + SAM2 detection & segmentation pipeline")
    parser.add_argument("--image", type=str, default="/app/assets/images/image1.jpg")
    parser.add_argument("--yolo-model", type=str, default="yolo/onnx/yolo11n.onnx")
    parser.add_argument("--sam2-model-dir", type=str, default="sam2/onnx/small")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--conf-threshold", type=float, default=0.45)
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--classes", type=parse_class_filter, default=None,
                        help="Filter YOLO classes (comma-separated IDs, e.g. 0,2,5)")
    args = parser.parse_args()

    main(args.image, args.yolo_model, args.sam2_model_dir, args.gpu,
         args.conf_threshold, args.iou_threshold, args.mask_threshold,
         args.classes)
