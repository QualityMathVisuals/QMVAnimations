ARG  APP_IMAGE=continuumio/miniconda3
FROM ${APP_IMAGE}

# Set working directory
WORKDIR /usr/src/app

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Update and install system dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libcairo2-dev \
    libpango1.0-dev \
    ffmpeg \
    ca-certificates \
    curl \
    pkg-config \
    redis-server \
    freeglut3-dev \
    xvfb \
    git \
    libsm6 \
    libxext6 \
    mesa-common-dev \
    texlive-full \
    dvisvgm \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Configure Conda and create environment
RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict && \
    pip install --upgrade pip && \
    pip install manimgl && \
    conda install -y sage==10.0

# There seems to be a numpy version error going on when sage is installed. This fixes it
RUN pip uninstall -y numpy && \
    pip uninstall -y numpy && \
    pip install numpy
