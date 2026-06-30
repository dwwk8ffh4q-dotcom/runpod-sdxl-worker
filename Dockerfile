FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y python3.10 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir -r requirements.txt

# Bake the public models into the image so cold starts don't re-download ~7GB.
# (The private style LoRA is small — 228MB — and is fetched at runtime via HF_TOKEN.)
RUN python3 -c "import torch; \
from diffusers import StableDiffusionXLPipeline; \
StableDiffusionXLPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16, variant='fp16', use_safetensors=True); \
from huggingface_hub import hf_hub_download; \
hf_hub_download('ByteDance/SDXL-Lightning', 'sdxl_lightning_4step_lora.safetensors')"

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
