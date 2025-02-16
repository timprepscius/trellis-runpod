# Use an official PyTorch image as the base image
FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

COPY requirements.txt ./

# Install system dependencies
RUN apt-get update
RUN apt-get install -y \
    git \
    python3-pip \
    cuda-compiler-12-0

RUN uname -m
RUN python3 --version

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --verbose -r requirements.txt


RUN git clone --recurse-submodules https://github.com/microsoft/TRELLIS
WORKDIR /app/TRELLIS

RUN apt install wget
RUN mkdir -p /root/.u2net/
RUN wget --no-verbose -O /root/.u2net/u2net.onnx "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"

RUN huggingface-cli download JeffreyXiang/TRELLIS-image-large

RUN mkdir -p  /root/.cache/torch/hub/checkpoints
RUN wget --no-verbose -O /root/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth"

#RUN rm -rf /var/lib/apt/lists/*

COPY *.py ./
COPY *.json ./

#RUN python3 rp_warmup.py

RUN apt-get install -y zip
RUN wget --no-verbose -O /root/.cache/torch/hub/fbm.zip "https://github.com/facebookresearch/dinov2/zipball/main"
RUN cd /root/.cache/torch/hub/ && unzip fbm.zip && mv facebookresearch-dinov2-* facebookresearch_dinov2_main

CMD [ "python3", "-u", "rp_handler.py" ]
#CMD [ "/bin/bash" ]
