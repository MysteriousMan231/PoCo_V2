#!/bin/bash
set -e

docker build . -t poco-vicos-demo:ubuntu18.04-cuda11.2.0-cudnn8 \
                --build-arg CUDA_VERSION=11.2.0-cudnn8 \
                --build-arg OS_VERSION=ubuntu18.04
