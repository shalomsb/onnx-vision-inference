#!/bin/bash
cd "${0%/*}"

function usage
{
    echo "usage: ./launch.sh [-b/-d/-r] [--cpu/--gpu]"
    echo "Choose action from:"
    echo "      -b [model] | Build/download ONNX models (model: yolo, da3, scrfd, sam2, gdino; default: all)"
    echo "      -d | Develop inside Docker container"
    echo "      -r [model] | Run inference pipeline (model: yolo, da3, scrfd, sam2, gdino; default: yolo)"
    echo "Choose runtime (optional, default: --cpu):"
    echo "      --cpu | Use CPU base image (ubuntu:22.04)"
    echo "      --gpu | Use GPU base image (nvcr.io/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04)"
}

ACTION=""
MODEL=""

# Auto-detect GPU
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    RUNTIME="gpu"
else
    RUNTIME="cpu"
fi

if [[ $# -lt 1 ]]; then
    usage && exit;
fi

while [[ "$1" != "" ]]; do
    case $1 in
        -b | -d | -r )  ACTION=$1   ;;
        --cpu )         RUNTIME=cpu ;;
        --gpu )         RUNTIME=gpu ;;
        -h )            usage && exit ;;
        * )
            if [[ ($ACTION == '-r' || $ACTION == '-b') && -z $MODEL ]]; then
                MODEL=$1
            else
                usage && exit
            fi
            ;;
    esac
    shift;
done

# Select base image
if [[ $RUNTIME == "gpu" ]]; then
    BASE_IMAGE="nvcr.io/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04"
else
    BASE_IMAGE="ubuntu:22.04"
fi

DOCKER_FILE=Dockerfile
DOCKER_TAG=onnx-vision-inference
DOCKER_TAG_VERSION=v1.0
DOCKER_NAME=onnx-vision-inference

echo "Runtime : $RUNTIME"
echo "Base    : $BASE_IMAGE"

if [[ $ACTION == '-b' ]]; then
    time \
    docker build -f $DOCKER_FILE \
        --build-arg BASE_IMAGE=${BASE_IMAGE} \
        -t $DOCKER_TAG:$DOCKER_TAG_VERSION .
fi

if [[ $ACTION == '-b' ]] || [[ $ACTION == '-r' ]] || [[ $ACTION == '-d' ]]; then

    DOCKER_ARGS="--name=${DOCKER_NAME} --rm --net=host --ipc=host --shm-size=4g -it"
    DOCKER_ARGS="$DOCKER_ARGS -v $(pwd)/../app:/app"

    if [[ $RUNTIME == "gpu" ]]; then
        DOCKER_ARGS="$DOCKER_ARGS --runtime=nvidia --gpus all"
    fi

    docker run $DOCKER_ARGS \
        --entrypoint /opt/entrypoint.sh \
        $DOCKER_TAG:$DOCKER_TAG_VERSION \
        $ACTION $MODEL;
    exit;

else
    usage
    exit 1
fi
