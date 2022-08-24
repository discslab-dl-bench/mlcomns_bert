FROM tensorflow/tensorflow:1.15.2-gpu

COPY . /workspace/bert
WORKDIR /workspace/bert

# Hack to avoid this error https://github.com/NVIDIA/nvidia-docker/issues/1632
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN apt-get install -y vim

RUN pip install --upgrade pip
RUN pip install --disable-pip-version-check -r requirements.txt
