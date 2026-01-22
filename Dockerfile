FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Системные зависимости (opencv и базовые утилиты)
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.8 python3.8-distutils python3-pip \
      git ca-certificates \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Делает `python` -> python3.8
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

WORKDIR /workspace

# Сначала зависимости (лучше для кеша)
COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel

# ВАЖНО:
# 1) ставим зависимости проекта
# 2) затем гарантируем PyTorch 2.0.0 (CUDA 11.7) по оф. инструкции
# 3) и (практично) заменяем TF на CPU-вариант, чтобы не упираться в CUDA-совместимость TF 2.6
RUN python -m pip install -r requirements.txt && \
    python -m pip uninstall -y tensorflow && \
    python -m pip install tensorflow-cpu==2.6.0 && \
    python -m pip uninstall -y torch torchvision torchaudio || true && \
    python -m pip install \
      torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 \
      --index-url https://download.pytorch.org/whl/cu118 \
      --extra-index-url https://pypi.org/simple


# Теперь код
COPY . /workspace

CMD ["bash"]
