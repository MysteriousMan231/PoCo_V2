ARG OS_VERSION=ubuntu:18.04
ARG CUDA_VERSION=11.2.0-cudnn8

FROM nvidia/cuda:${CUDA_VERSION}-runtime-${OS_VERSION}

LABEL maintainer "pm4824@student.uni-lj.si"

ENV DEBIAN_FRONTEND noninteractive

######################################
# install dependencies for vicos-demo (echolib and echocv)

ENV MODEL_DIR /opt

WORKDIR $MODEL_DIR

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git python3-dev python3-numpy-dev python3-pip libopencv-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# install dependency pybind11
RUN git clone --depth 1 https://github.com/wjakob/pybind11.git && cd pybind11 && mkdir build && cd build && \
    cmake -j -DPYBIND11_TEST=OFF -DPYBIND11_INSTALL=ON .. && make -j install && cd ../.. && rm -r pybind11

# install echolib
RUN git clone --depth 1 https://github.com/vicoslab/echolib.git
RUN cd ${MODEL_DIR}//echolib && mkdir build && cd build && \
    cmake -DBUILD_DAEMON=OFF .. && make -j && make install && cd ../.. && rm -r echolib

# install echocv
RUN git clone --depth 1 https://github.com/vicoslab/echocv.git
RUN cd ${MODEL_DIR}//echocv && mkdir build && cd build && \
    cmake -DBUILD_APPS=OFF .. && make -j && make install && cd ../.. && rm -r echocv

##################################
# install dependencies for poco-demo
RUN python3 -m pip install --upgrade pip && python3 -m pip install setuptools
RUN python3 -m pip install tensorflow==2.6.0 opencv-python>=4

COPY scripts ${MODEL_DIR}

RUN chmod +x ${MODEL_DIR}/run_main.py

# define entry-point and default arguments
ENTRYPOINT ["/opt/run_main.py"]