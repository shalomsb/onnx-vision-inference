import argparse
import time

import numpy as np
import cv2

from .preprocess import preprocess, transform_boxes, transform_coords
from .inference import load_models, run_encoder, run_decoder
from .postprocess import postprocess
from .visualize import save_results


def build_prompts(boxes, points, scale_x, scale_y):
    """
    Build point_coords and point_labels arrays from CLI box/point arguments.

    Args:
        boxes:  list of [x1, y1, x2, y2] arrays (original image coords)
        points: list of [x, y, label] arrays (original image coords)
        scale_x, scale_y: scaling factors to 1024x1024

    Returns:
        point_coords: [N, num_pts, 2] float32
        point_labels: [N, num_pts] float32
    """
    all_coords = []
    all_labels = []

    if boxes is not None and len(boxes) > 0:
        box_arr = np.array(boxes, dtype=np.float32)
        coords, labels = transform_boxes(box_arr, scale_x, scale_y)
        # Each box is one prompt: [1, 2, 2] coords, [1, 2] labels
        for i in range(len(box_arr)):
            all_coords.append(coords[i])   # [2, 2]
            all_labels.append(labels[i])    # [2]

    if points is not None and len(points) > 0:
        # Points without boxes: each point is its own prompt
        # Points with boxes: append to each box prompt (not implemented for simplicity)
        for pt in points:
            x, y, label = pt
            scaled = transform_coords(np.array([[x, y]], dtype=np.float32), scale_x, scale_y)
            all_coords.append(scaled)                          # [1, 2]
            all_labels.append(np.array([label], dtype=np.float32))  # [1]

    if not all_coords:
        raise ValueError("At least one --box or --point prompt is required")

    # Pad all prompts to same number of points (max across prompts)
    max_pts = max(c.shape[0] for c in all_coords)
    n = len(all_coords)
    point_coords = np.zeros((n, max_pts, 2), dtype=np.float32)
    point_labels = np.full((n, max_pts), -1.0, dtype=np.float32)  # -1 = padding

    for i in range(n):
        num_pts = all_coords[i].shape[0]
        point_coords[i, :num_pts] = all_coords[i]
        point_labels[i, :num_pts] = all_labels[i]

    return point_coords, point_labels


def main(image_path: str, model_dir: str = "sam2/onnx/small",
         gpu: bool = False, boxes: list = None, points: list = None,
         mask_threshold: float = 0.5):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    orig_h, orig_w = image.shape[:2]
    print(f"Image: {image_path} ({orig_w}x{orig_h})")

    # Preprocess
    blob, scale_x, scale_y = preprocess(image)

    # Load models
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
    print(f"Using ONNXRuntime providers: {providers}")
    sessions = load_models(model_dir, providers=providers)

    # Encode image
    start = time.perf_counter()
    features = run_encoder(sessions["encoder"], blob)
    encode_ms = (time.perf_counter() - start) * 1000
    print(f"Encoder time: {encode_ms:.2f} ms")

    # Build prompts
    point_coords, point_labels = build_prompts(boxes, points, scale_x, scale_y)
    n = point_coords.shape[0]
    print(f"Prompts: {n} ({len(boxes or [])} boxes, {len(points or [])} points)")

    # Expand image_embed for batch of prompts
    image_embed = np.tile(features["pix_feat"], (n, 1, 1, 1))  # [N, 256, 64, 64]

    # Decode masks
    start = time.perf_counter()
    pred_masks, iou_scores = run_decoder(
        sessions["decoder"],
        point_coords,
        point_labels,
        image_embed,
        features["high_res_feat0"],
        features["high_res_feat1"],
    )
    decode_ms = (time.perf_counter() - start) * 1000
    print(f"Decoder time: {decode_ms:.2f} ms")

    # Postprocess
    result = postprocess(pred_masks, iou_scores, orig_h, orig_w,
                         mask_threshold=mask_threshold)

    num_masks = len(result["masks"])
    print(f"\n{num_masks} mask(s) generated")
    for i, score in enumerate(result["scores"]):
        mask_area = result["masks"][i].sum()
        print(f"  mask {i}: IoU={score:.3f}, area={mask_area} px")

    # Build prompts dict for visualization
    vis_prompts = {}
    if boxes:
        vis_prompts["boxes"] = np.array(boxes, dtype=np.float32)
    if points:
        vis_prompts["points"] = np.array(points, dtype=np.float32)

    save_results([image], [result],
                 prompts=[vis_prompts] if vis_prompts else None)


def parse_box(s: str) -> list[float]:
    """Parse 'x1,y1,x2,y2' string into [x1, y1, x2, y2]."""
    parts = s.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"Box must be x1,y1,x2,y2 — got: {s}")
    return [float(x) for x in parts]


def parse_point(s: str) -> list[float]:
    """Parse 'x,y,label' string into [x, y, label]."""
    parts = s.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Point must be x,y,label — got: {s}")
    return [float(x) for x in parts]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2 image segmentation")
    parser.add_argument("--image", type=str, default="/app/assets/images/image1.jpg")
    parser.add_argument("--model-dir", type=str, default="sam2/onnx/small")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--box", type=parse_box, action="append", dest="boxes",
                        help="Bounding box prompt: x1,y1,x2,y2 (repeatable)")
    parser.add_argument("--point", type=parse_point, action="append", dest="points",
                        help="Point prompt: x,y,label (1=fg, 0=bg) (repeatable)")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    args = parser.parse_args()

    main(args.image, args.model_dir, args.gpu, args.boxes, args.points,
         args.mask_threshold)
