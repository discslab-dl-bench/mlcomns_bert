FROM tensorflow/tensorflow:1.15.2-gpu

# Hack to avoid this error https://github.com/NVIDIA/nvidia-docker/issues/1632
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git vim wget libtool autoconf build-essential

# # Install Darshan
# WORKDIR /opt
# RUN wget https://ftp.mcs.anl.gov/pub/darshan/releases/darshan-3.4.0.tar.gz
# RUN tar -xzf darshan-3.4.0.tar.gz
# WORKDIR darshan-3.4.0
# RUN ./prepare.sh
# WORKDIR darshan-runtime/
# RUN ./configure --with-log-path-by-env=DARSHAN_LOGPATH --with-jobid-env=NONE --without-mpi --enable-mmap-logs CC=gcc
# RUN make
# RUN make install

COPY . /workspace/bert
WORKDIR /workspace/bert


RUN pip install --upgrade pip
RUN pip install --disable-pip-version-check -r requirements.txt
