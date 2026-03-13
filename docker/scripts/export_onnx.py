import argparse
import os
from ultralytics import YOLO


def export(version: str, size: str):
    if version == "11":
        model_name = f"yolo{version}{size}.pt"
    else:
        model_name = f"yolov{version}{size}.pt"
    onnx_name = model_name.replace(".pt", ".onnx")

    model = YOLO(model_name)
    exported = model.export(
        format="onnx",
        imgsz=640,
        opset=12,
        simplify=True,
        dynamic=True,
    )

    os.makedirs("onnx", exist_ok=True)
    os.rename(exported, os.path.join("onnx", onnx_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="11", choices=["8", "9", "11"])
    parser.add_argument("--size", type=str, default="n", choices=["n", "s", "m", "l", "x"])
    args = parser.parse_args()

    export(args.version, args.size)
