FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install python3.13
RUN : \
    && apt-get update \
    && apt-get install -y git \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get install -y --no-install-recommends python3.13-venv \
    && apt-get install libpython3.13-dev -y \
    && apt-get clean \
    && :

# Add env to PATH
RUN python3.13 -m venv /venv
ENV PATH=/venv/bin:$PATH

# Install OpenSlide dependencies
RUN : \
    && apt-get update \
    && apt-get install -y openslide-tools libopenslide0 \
    && apt-get install -y build-essential libffi-dev libxml2-dev libjpeg-turbo8-dev zlib1g-dev \
    && apt-get clean \
    && : \

# Install OpenCV dependencies
RUN : \
    && apt-get update \
    && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    && apt-get clean \
    && :

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

# update permissions
RUN chown -R user:user /venv/

USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/

# Update pip
RUN /venv/bin/python3.13 -m pip install pip --upgrade

#install pytorch
RUN /venv/bin/python3.13 -m pip install torch~=2.6.0 torchvision~=0.21.0 torchaudio~=2.6.0 -f https://download.pytorch.org/whl/cu122/torch_stable.html

# You can add any Python dependencies to requirements.txt
RUN /venv/bin/python3.13 -m pip install \
    --no-cache-dir \
    -r /opt/app/requirements.txt

# Verify torch installation to ensure it's available
RUN /venv/bin/python3.13 -c "import torch; print(torch.__version__)"

COPY --chown=user:user inference.py /opt/app/

USER user
ENTRYPOINT ["/venv/bin/python3.13", "inference.py"]
