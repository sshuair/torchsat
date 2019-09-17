ARG UBUNTU_VERSION=18.04

ARG CUDA=10.0
ARG CUDNN=7
FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu${UBUNTU_VERSION} as base 

RUN apt-get update && apt-get install -y python3-pip

RUN pip3 --no-cache-dir install --upgrade \
    pip \
    setuptools

RUN pip3 --no-cache-dir install \
    numpy>=1.13.0 \
    scipy>=1.1.0 \
    pillow>=5.0.0 \ 
    six>=1.11.0 \
    tifffile>=2019.1.1 \
    opencv-python>=3.2.0.6 \
    torch>=1.0.0 \
    torchvision>=0.2.0

COPY ./ /tmp

RUN cd /tmp && python3 setup.py install

WORKDIR /workspace