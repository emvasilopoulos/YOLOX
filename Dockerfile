# base image nvcr.io/nvidia/cuda:12.1.1-devel-ubuntu22.04
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch?version=25.12-py3
FROM nvcr.io/nvidia/pytorch:25.12-py3-igpu

# Setup OS environment
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install dependencies
RUN apt-get update && apt-get install -y \
    git wget curl ca-certificates build-essential cmake \
    python3 python3-pip python3-setuptools python3-venv python3-dev \
    libgl1 libglib2.0-0 tmux \
    groff \ # for AWS CLI help command
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install numpy==1.26.3 opencv-python==4.8.0.74 opencv-contrib-python==4.8.0.74 
RUN pip3 install loguru==0.7.3 scikit-image==0.25.2 scipy==1.15.3 \
    tqdm==4.67.1 \
    Pillow==12.0.0 \
    thop==0.1.1.post2209072238 \
    ninja==1.13.0 \
    tabulate==0.9.0 \
    tensorboard==2.20.0 \
    pycocotools==2.0.11
RUN git clone https://github.com/emvasilopoulos/YOLOX.git /YOLOX && cd /YOLOX && pip3 install . --no-build-isolation

# To download OpenImages
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip -q && \
    ./aws/install && \
    rm -rf awscliv2.zip aws


WORKDIR /home/

CMD ["/bin/bash"]