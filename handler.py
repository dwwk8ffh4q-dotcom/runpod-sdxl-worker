import runpod
import torch
import base64
import io
import os
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import login

pipe = None
MODEL_ID = os.environ.get("MODEL_ID", "stabilityai/stable-diffusion-3.5-large")


def load_pipeline():
    global pipe

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    print(f"Loading {MODEL_ID} (bf16)...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    # Full bf16 quality on a 24GB GPU: offload idle modules (T5, etc.) to CPU.
    # Slower per image than a fully-resident model, but fits and keeps quality.
    pipe.enable_model_cpu_offload()
    print("Pipeline ready.")


def generate(job):
    global pipe

    if pipe is None:
        load_pipeline()

    inp = job["input"]
    prompt = inp.get("prompt", "")
    negative_prompt = inp.get("negative_prompt", "")
    width = int(inp.get("width", 1024))
    height = int(inp.get("height", 576))
    seed = int(inp.get("seed", -1))
    steps = int(inp.get("steps", 30))
    guidance = float(inp.get("guidance_scale", 4.5))

    generator = None
    if seed != -1:
        # cpu_offload keeps the generator on CPU
        generator = torch.Generator("cpu").manual_seed(seed)

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            max_sequence_length=512,  # let SD3.5 read the full detailed prompt
            generator=generator,
        )

    image = result.images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {"image": f"data:image/png;base64,{img_b64}"}


# Preload at worker startup so the (one-time) model load counts as init, not per-request.
load_pipeline()

runpod.serverless.start({"handler": generate})
