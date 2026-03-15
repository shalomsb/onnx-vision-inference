#!/usr/bin/env bash
# Download SCRFD 10G ONNX model from InsightFace buffalo_l pack

ONNX_DIR="/app/scrfd/onnx"
ONNX_FILE="$ONNX_DIR/det_10g.onnx"
URL="https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_10g.onnx"

if [[ -f "$ONNX_FILE" ]]; then
    echo "SCRFD ONNX model already exists: $ONNX_FILE — skipping download."
    exit 0
fi

mkdir -p "$ONNX_DIR"

echo "Downloading det_10g.onnx (~17 MB) ..."
wget -q --show-progress -O "$ONNX_FILE" "$URL"

if [[ $? -eq 0 && -f "$ONNX_FILE" ]]; then
    echo "Download complete: $ONNX_FILE"
else
    echo "ERROR: Failed to download SCRFD ONNX model."
    rm -f "$ONNX_FILE"
    exit 1
fi
