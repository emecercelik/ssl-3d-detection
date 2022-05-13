## non-root user
FROM tensorflow/tensorflow:1.15.2-gpu-py3

ARG DEBIAN_FRONTEND=noninteractive

## Install requirements
RUN apt-get update && apt-get install -y git nano protobuf-compiler python-pil python-lxml python-tk

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

## Clone models
RUN git clone https://github.com/tensorflow/models.git /usr/local/lib/python3.6/dist-packages/tensorflow_core/models
WORKDIR /usr/local/lib/python3.6/dist-packages/tensorflow_core/models/research

## Install protoc
RUN curl -OL https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip && \
	unzip protoc-3.3.0-linux-x86_64.zip -d protoc3 &&\
	mv protoc3/bin/* /usr/local/bin/ &&\
	mv protoc3/include/* /usr/local/include/ &&\
	protoc object_detection/protos/*.proto --python_out=.

######## Install OpenCV #######
RUN  apt-get update && apt-get install -y build-essential apt-utils cmake libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy python3 python3-pip python3-dev libtbb2 libtbb-dev libjpeg-dev libdc1394-22-dev python-opencv libopencv-dev  python-pycurl libatlas-base-dev gfortran webp qt5-default libvtk6-dev zlib1g-dev

RUN cd ~/ &&\
    git clone https://github.com/Itseez/opencv.git &&\
    git clone https://github.com/Itseez/opencv_contrib.git &&\
    cd opencv && mkdir build && cd build && cmake  -DWITH_QT=ON -DWITH_OPENGL=ON -DFORCE_VTK=ON -DWITH_TBB=ON -DWITH_GDAL=ON -DWITH_XINE=ON -DBUILD_EXAMPLES=ON .. && \
    make -j4 && make install && ldconfig && rm -r /root/opencv && rm -r /root/opencv_contrib && ln /dev/null /dev/raw1394

WORKDIR /

# For eval functions
RUN  apt-get update && apt-get install -y gnuplot gnuplot5 ghostscript texlive-extra-utils

## Non-root operation
RUN useradd --user-group --create-home --no-log-init --shell /bin/bash selfsup_pointgnn
USER selfsup_pointgnn

# Show path to the Tensorflow Object Detection API
RUN echo "export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.6/dist-packages/tensorflow_core/models/research/:/usr/local/lib/python3.6/dist-packages/tensorflow_core/models/research/slim" >> ~/.bashrc

