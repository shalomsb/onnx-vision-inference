import argparse
import time

import cv2

from .preprocess import preprocess
from .inference import load_model, run_inference
from .postprocess import postprocess
from .visualize import draw_detections, save_results


def main(image_path: str, model_path: str = "scrfd/onnx/det_10g.onnx",
         input_size: int = 640, gpu: bool = False,
         conf_threshold: float = 0.5, iou_threshold: float = 0.4):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    blob, scale, pad = preprocess(image, (input_size, input_size))

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
    print(f"Using ONNXRuntime providers: {providers}")
    session = load_model(model_path, providers=providers)

    start = time.perf_counter()
    outputs = run_inference(session, blob)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"Inference time: {elapsed:.2f} ms")

    result = postprocess(outputs, input_size, scale, pad,
                         conf_threshold=conf_threshold,
                         iou_threshold=iou_threshold)

    num_faces = len(result["boxes"])
    print(f"\n{num_faces} face(s) detected")
    for i, (box, score) in enumerate(zip(result["boxes"], result["scores"])):
        print(f"  face {i}: conf={score:.2f}, box={box.astype(int).tolist()}")

    save_results([image], [result])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCRFD 10G face detection")
    parser.add_argument("--image", type=str, default="/app/assets/images/image1.jpg")
    parser.add_argument("--model", type=str, default="scrfd/onnx/det_10g.onnx")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.4)
    args = parser.parse_args()

    main(args.image, args.model, args.input_size, args.gpu,
         args.conf_threshold, args.iou_threshold)
