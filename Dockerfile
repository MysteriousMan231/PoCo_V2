ARG OS_VERSION=ubuntu:16.04
ARG DETECTRON_IMAGE_RUNTIME

FROM ${OS_VERSION} as model-dl

RUN apt-get update && apt-get install -y unzip

ENV MODEL_DIR /opt

WORKDIR $MODEL_DIR

FROM ${DETECTRON_IMAGE_RUNTIME}

LABEL maintainer "pm4824@student.uni-lj.si"

######################################
# install dependencies for vicos-demo (echolib and echocv)

ENV MODEL_DIR /opt

WORKDIR $MODEL_DIR

RUN apt-get update && apt-get install -y build-essential cmake libopencv-dev python-numpy-dev 

# install dependency pybind11
RUN git clone https://github.com/wjakob/pybind11.git && cd pybind11 && mkdir build && cd build && cmake -DPYBIND11_TEST=OFF -DPYBIND11_INSTALL=ON .. && make -j install

# install echolib (use version compatible with python2.7)
RUN git clone https://github.com/vicoslab/echolib.git && cd ${MODEL_DIR}/echolib && git checkout f80d4dd71f7def76880e8f3ea915978ce56efcfb
RUN cd ${MODEL_DIR}//echolib && mkdir build && cd build && cmake -DBUILD_DAEMON=OFF .. && make -j && make install && cd .. && rm -r build

# install echocv (use version compatible with python2.7)
RUN git clone https://github.com/vicoslab/echocv.git && cd ${MODEL_DIR}//echocv && git checkout 5a911db977676758f308193f7a13aa7875a7d114
RUN cd ${MODEL_DIR}//echocv && mkdir build && cd build && cmake -DBUILD_APPS=OFF .. && make -j && make install && cd .. && rm -r build

RUN pip install tensorflow==1.14.0
RUN pip install keras==2.3.1
RUN pip install opencv-python==4.1.1.26

##################################
# install traffic sign model and scripts

COPY scripts ${MODEL_DIR}

RUN chmod +x ${MODEL_DIR}/run_main.py

# define entry-point and default arguments
ENTRYPOINT ["/opt/run_main.py"]