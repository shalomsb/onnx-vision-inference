#!/usr/bin/env bash
# Download DA3METRIC-LARGE ONNX model from HuggingFace

ONNX_DIR="/app/da3/onnx"
ONNX_FILE="$ONNX_DIR/DA3METRIC-LARGE.onnx"
URL="https://huggingface.co/TillBeemelmanns/Depth-Anything-V3-ONNX/resolve/main/DA3METRIC-LARGE.onnx"

if [[ -f "$ONNX_FILE" ]]; then
    echo "DA3 ONNX model already exists: $ONNX_FILE — skipping download."
    exit 0
fi

mkdir -p "$ONNX_DIR"

echo "Downloading DA3METRIC-LARGE.onnx (~731 MB) ..."
wget -q --show-progress -O "$ONNX_FILE" "$URL"

if [[ $? -eq 0 && -f "$ONNX_FILE" ]]; then
    echo "Download complete: $ONNX_FILE"
else
    echo "ERROR: Failed to download DA3 ONNX model."
    rm -f "$ONNX_FILE"
    exit 1
fi
