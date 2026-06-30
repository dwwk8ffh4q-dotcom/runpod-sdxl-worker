import runpod
import torch
import base64
import io
import os
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from huggingface_hub import login

pipe = None


def load_pipeline():
    global pipe

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    print("Loading SDXL Base 1.0...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        add_watermarker=False,
    ).to("cuda")

    print("Loading SDXL Lightning 4-step LoRA...")
    pipe.load_lora_weights(
        "ByteDance/SDXL-Lightning",
        weight_name="sdxl_lightning_4step_lora.safetensors",
        adapter_name="lightning",
    )

    print("Loading style LoRA (ytanim)...")
    pipe.load_lora_weights(
        "fedcsx/youtube-style-lora",
        weight_name="youtube_style.safetensors",
        adapter_name="style",
    )

    # Lightning at full weight, style LoRA at 0.8
    pipe.set_adapters(["lightning", "style"], adapter_weights=[1.0, 0.8])

    # Lightning requires trailing timestep spacing
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
    )

    pipe.unet.to(memory_format=torch.channels_last)
    print("Pipeline ready.")


def generate(job):
    global pipe

    if pipe is None:
        load_pipeline()

    inp = job["input"]
    prompt = inp.get("prompt", "")
    negative_prompt = inp.get(
        "negative_prompt",
        "photorealistic, photo, 3d render, 3d cgi, anime shading, gradient fills, "
        "nsfw, text, watermark, ugly, deformed, blurry, low quality, realistic lighting",
    )
    width = int(inp.get("width", 768))
    height = int(inp.get("height", 448))
    seed = int(inp.get("seed", -1))
    steps = int(inp.get("steps", 4))

    generator = None
    if seed != -1:
        generator = torch.Generator("cuda").manual_seed(seed)

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=0,  # Lightning requires CFG=0
            generator=generator,
        )

    image = result.images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {"image": f"data:image/png;base64,{img_b64}"}


runpod.serverless.start({"handler": generate})
