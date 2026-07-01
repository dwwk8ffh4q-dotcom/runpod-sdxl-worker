FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Cache the model on a persistent network volume mounted at /runpod-volume so the
# ~20GB SD3.5 download happens once, not on every cold start.
ENV HF_HOME=/runpod-volume/huggingface

RUN apt-get update && apt-get install -y python3.10 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir -r requirements.txt

# SD3.5 is a gated model, so it can't be baked at build time without a build secret.
# It downloads at first worker startup using the runtime HF_TOKEN, into HF_HOME
# (the network volume), so it persists across workers.

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
