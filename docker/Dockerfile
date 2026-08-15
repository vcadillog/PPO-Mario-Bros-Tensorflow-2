FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py37_23.1.0-1-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda create -y -n mario python=3.6
ENV PATH=$CONDA_DIR/envs/mario/bin:$PATH
SHELL ["conda", "run", "-n", "mario", "/bin/bash", "-c"]

RUN conda install -y -n mario -c anaconda cudatoolkit=10.0 cudnn=7.6.5

RUN python -m pip install --upgrade pip setuptools wheel

RUN python -u -m pip install --use-deprecated=legacy-resolver numpy==1.17.3
RUN python -u -m pip install --use-deprecated=legacy-resolver six==1.12.0 future==0.16.0 cloudpickle==1.2.2 tqdm==4.28.1
RUN python -u -m pip install --use-deprecated=legacy-resolver scipy==1.3.1
RUN python -u -m pip install --use-deprecated=legacy-resolver pyglet==1.3.2
RUN python -u -m pip install --use-deprecated=legacy-resolver gym==0.15.3
RUN python -u -m pip install --use-deprecated=legacy-resolver opencv-python==4.5.5.64
RUN python -u -m pip install --use-deprecated=legacy-resolver tensorflow-gpu==2.0.0
RUN python -u -m pip install --use-deprecated=legacy-resolver nes-py==8.1.1
RUN python -u -m pip install --use-deprecated=legacy-resolver gym-super-mario-bros==7.3.0

WORKDIR /workspace
RUN git clone https://github.com/vcadillog/PPO-Mario-Bros-Tensorflow-2.git

WORKDIR /workspace/PPO-Mario-Bros-Tensorflow-2

EXPOSE 6006

CMD ["conda", "run", "--no-capture-output", "-n", "mario", "/bin/bash"]
