#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" 2>/dev/null && pwd || pwd)"
cd "$SCRIPT_DIR"

function usage
{
    echo "usage: ./export_onnx.sh [--version 8/9/11] [--size n/s/m/l/x]"
    echo "default version: 11, default size: n"
    echo "for help: ./export_onnx.sh -h"
}

VERSION=11
SIZE=n

if [[ $# -gt 4 ]]; then
    usage && exit;
fi

while [[ "$1" != "" ]]; do
    case $1 in
        --version )    shift; VERSION=$1 ;;
        --size )       shift; SIZE=$1    ;;
        -h )           usage && exit ;;
        * )            usage && exit ;;
    esac
    shift;
done

# Check if version is valid
if [[ $VERSION != "8" && $VERSION != "9" && $VERSION != "11" ]]; then
    echo "Invalid version: $VERSION"
    usage && exit;
fi

# Check if size is valid
if [[ $SIZE != "n" && $SIZE != "s" && $SIZE != "m" && $SIZE != "l" && $SIZE != "x" ]]; then
    echo "Invalid size: $SIZE"
    usage && exit;
fi

# Build model name
if [[ $VERSION == "11" ]]; then
    MODEL_NAME="yolo${VERSION}${SIZE}"
else
    MODEL_NAME="yolov${VERSION}${SIZE}"
fi

# Check if onnx already exists
if [[ -f "/app/yolo/onnx/${MODEL_NAME}.onnx" ]]; then
    echo "ONNX model already exists: /app/yolo/onnx/${MODEL_NAME}.onnx"
    exit;
fi

python3 -c "import ultralytics" 2>/dev/null || python3 -m pip install ultralytics

python3 export_onnx.py --version $VERSION --size $SIZE

# Restore onnxruntime-gpu (ultralytics may overwrite it with CPU-only onnxruntime)
pip3 install --no-cache-dir onnxruntime-gpu

mkdir -p /app/yolo/onnx
mv ./onnx/${MODEL_NAME}.onnx /app/yolo/onnx/${MODEL_NAME}.onnx

