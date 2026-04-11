import argparse
import time

import cv2

from .preprocess import preprocess_image, load_tokenizer, preprocess_text, map_token_to_category
from .inference import load_model, run_inference
from .postprocess import postprocess
from .visualize import save_results


def main(image_path: str,
         model_path: str = "groundingdino/onnx/model.onnx",
         text: str = "object",
         gpu: bool = False,
         score_thresh: float = 0.30,
         iou_thresh: float = 0.50):

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    orig_h, orig_w = image.shape[:2]
    print(f"Image: {image_path} ({orig_w}x{orig_h})")

    # Preprocess image
    blob, pixel_mask, orig_size = preprocess_image(image)

    # Preprocess text
    print(f"Text prompt: \"{text}\"")
    tokenizer = load_tokenizer()
    text_inputs = preprocess_text(tokenizer, text)

    # Load model
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
    print(f"Using ONNXRuntime providers: {providers}")
    session = load_model(model_path, providers=providers)

    # Inference
    start = time.perf_counter()
    pred_logits, pred_boxes = run_inference(session, blob, pixel_mask, text_inputs)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"Inference time: {elapsed:.2f} ms")

    # Postprocess (per-category NMS)
    result = postprocess(pred_logits, pred_boxes, orig_size,
                         score_thresh=score_thresh, iou_thresh=iou_thresh)

    # Map token IDs to category names
    det_labels = map_token_to_category(tokenizer, text, result["token_ids"])

    num_det = len(result["boxes"])
    print(f"\n{num_det} detection(s)  prompt=\"{text}\"")
    for i, (box, score, lbl) in enumerate(zip(result["boxes"], result["scores"], det_labels)):
        print(f"  [{i}] {lbl} score={score:.3f} box={box.astype(int).tolist()}")

    save_results([image], [result], labels=[det_labels])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grounding DINO open-set object detection")
    parser.add_argument("--image", type=str, default="/app/assets/images/image1.jpg")
    parser.add_argument("--model", type=str,
                        default="groundingdino/onnx/model.onnx")
    parser.add_argument("--text", type=str, required=True,
                        help='Grounding prompt, e.g. "person . car ."')
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--score-thresh", type=float, default=0.30)
    parser.add_argument("--iou-thresh", type=float, default=0.50)
    args = parser.parse_args()

    main(args.image, args.model, args.text, args.gpu,
         args.score_thresh, args.iou_thresh)
