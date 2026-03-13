# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ONNX-based vision inference project that runs vision models (YOLO, DepthPro/DP3, SAM3) using ONNX Runtime with CPU/GPU support inside Docker.

## Architecture

- **`Dockerfile/`** — Docker build context: Dockerfile, requirements.txt, launch.sh (host-side), entrypoint.sh (container-side)
- **`app/`** — Model-specific code organized by model name:
  - `yolo/` — YOLO11 object detection (export script converts PyTorch `.pt` to ONNX)
  - `dp3/` — DepthPro (depth estimation) — placeholder
  - `sam3/` — SAM3 (segmentation) — placeholder
- At runtime, the container mounts `models/`, `pipelines/`, `utils/`, and `scripts/` from the host into `/app` and `/opt`

## Docker Workflow

All development and inference runs inside Docker. The `Dockerfile/launch.sh` script is the primary entry point:

```bash
# Build (from repo root — launch.sh cd's to its own dir)
./Dockerfile/launch.sh -b [--cpu|--gpu]

# Develop (interactive bash inside container)
./Dockerfile/launch.sh -d [--cpu|--gpu]

# Run inference
./Dockerfile/launch.sh -r [--cpu|--gpu]
```

- `--cpu` (default): base image `ubuntu:22.04`
- `--gpu`: base image `nvcr.io/nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04`, adds `--runtime=nvidia --gpus all`
- Image tag: `onnx-vision-inference:v1.0`

## Key Dependencies

Python 3.11 inside the container. Core packages: `onnxruntime-gpu`, `numpy 1.26.0`, `opencv-python-headless`, `PyYAML`. Model export uses `ultralytics` (not a runtime dependency).

## ONNX Export

Models are exported to ONNX format before inference. Example for YOLO11:
```bash
python3 app/yolo/export_onnx.py
```
Uses opset 12, static batch=1, 640×640 input, with onnx-simplifier.
