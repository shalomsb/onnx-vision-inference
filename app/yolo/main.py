import argparse
import time

from .dataloader import load_images, create_batch
from .inference import load_model, run_inference
from .postprocess import postprocess
from .visualize import save_results


def main(images_dir: str = None, image: str = None, model_path: str = "yolo/onnx/yolo11n.onnx", gpu: bool = False):
    images = load_images(images_dir, image)
    if not images:
        print("No images found.")
        return

    batch, scales, pads = create_batch(images)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
    print(f"Using ONNXRuntime providers: {providers}")
    session = load_model(model_path, providers=providers)
    # print("warmup...")
    # for _ in range(10):
    #      run_inference(session, batch)
    start = time.perf_counter()
    output = run_inference(session, batch)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"Inference time: {elapsed:.2f} ms")
    print(f"Output shape: {output.shape}")

    results = postprocess(output, scales, pads, conf_threshold=0.45, iou_threshold=0.45)
    for i, det in enumerate(results):
        print(f"\nImage {i}: {len(det['boxes'])} detections")
        for box, score, cls_id in zip(det["boxes"], det["scores"], det["class_ids"]):
            print(f"  class={int(cls_id)}, conf={score:.2f}, box={box.astype(int).tolist()}")

    save_results(images, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="../assets/images/image1.jpg")
    parser.add_argument("--images_dir", type=str, default=None)
    parser.add_argument("--model", type=str, default="yolo/onnx/yolo11n.onnx")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    main(args.images_dir, args.image, args.model, args.gpu)
