FROM tensorflow/tensorflow:1.15.2-gpu

# Hack to avoid this error https://github.com/NVIDIA/nvidia-docker/issues/1632
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git vim wget libtool autoconf build-essential mpich 
    
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt-get update && \
    apt-get install libnccl2=2.8.4-1+cuda10.2 libnccl-dev=2.8.4-1+cuda10.2

COPY . /workspace/bert
WORKDIR /workspace/bert

RUN pip install --upgrade pip
RUN pip install --disable-pip-version-check -r requirements.txt

RUN HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 pip install --no-cache-dir horovod
