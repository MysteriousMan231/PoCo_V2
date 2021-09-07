#!/bin/bash
set -e

docker build . -t poco-vicos-demo:ubuntu16.04-cuda11.1-cudnn8 \
                --build-arg OS_VERSION=ubuntu:16.04 \
                --build-arg DETECTRON_IMAGE_RUNTIME=detectron-traffic-signs:ubuntu16.04-cuda11.1-cudnn8
