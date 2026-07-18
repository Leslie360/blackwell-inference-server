# Blackwell Inference Server
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY examples ./examples
COPY docs ./docs

# Install PyTorch with CUDA 13.0 support
RUN pip install --upgrade pip && \
    pip install torch triton --index-url https://download.pytorch.org/whl/cu130

# Install package
RUN pip install -e .[asr,server]

# Default to inference server
EXPOSE 8000
ENTRYPOINT ["blackwell-serve"]
CMD ["--help"]
