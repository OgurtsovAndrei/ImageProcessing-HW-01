import torch
from diffusers import FluxPipeline
import os

BASE_MODEL = "black-forest-labs/FLUX.1-dev"
OUTPUT_DIR = "generated_logos_flux"
PROMPTS = ["Telegram Computer Vision Group Logo, professional, modern design, vector style"]
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 3.5
NUM_IMAGES_PER_PROMPT = 1
HEIGHT = 1024
WIDTH = 1024

# LoRA settings
LORA_PATH = None  # Path to LoRA weights, e.g., "output_flux/best_lora" or None to disable
LORA_SCALE = 1.0  # LoRA strength (0.0 to 1.0+), higher = stronger effect

def load_flux_pipeline(model_id, device="cpu", lora_path=None, lora_scale=1.0):
    """Load Flux.1 pipeline with optional LoRA adapter.
    
    Args:
        model_id: HuggingFace model ID
        device: Device to load model on
        lora_path: Path to LoRA weights directory (None to disable)
        lora_scale: LoRA strength multiplier (0.0 to 1.0+)
    """
    print(f"Loading Flux.1 model: {model_id}")

    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,  # Flux prefers bf16, todo: can my MPS handle it?
    )

    pipe.enable_attention_slicing() # Used for memory optimizations

    pipe = pipe.to(device)
    print("Flux.1 pipeline loaded!")
    
    # Load LoRA adapter if path is provided
    if lora_path is not None and os.path.exists(lora_path):
        print(f"🎨 Loading LoRA adapter from: {lora_path}")
        print(f"   LoRA scale: {lora_scale}")
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_scale=lora_scale)
        print("✅ LoRA adapter loaded and fused!")
    elif lora_path is not None:
        print(f"⚠️  Warning: LoRA path specified but not found: {lora_path}")
        print("   Continuing without LoRA...")
    
    return pipe


def generate_images(pipe, prompts, output_dir, **kwargs):
    """Generate images with Flux.1."""
    os.makedirs(output_dir, exist_ok=True)

    for i, prompt in enumerate(prompts):
        print(f"\n Generating logo {i + 1}/{len(prompts)}")
        print(f"   Prompt: {prompt}")

        with torch.no_grad():
            default = NUM_INFERENCE_STEPS
            image = pipe(
                prompt=prompt,
                height=HEIGHT,
                width=WIDTH,
                num_inference_steps=kwargs.get('num_inference_steps', default),
                guidance_scale=kwargs.get('guidance_scale', GUIDANCE_SCALE),
            ).images[0]

        filename = f"logo_flux_{i + 1}.png"
        filepath = os.path.join(output_dir, filename)
        image.save(filepath, quality=95)
        print(f"   ✅ Saved: {filepath}")

    print(f"\n✨ All logos saved to: {output_dir}")


def main():
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")

    pipe = load_flux_pipeline(
        model_id=BASE_MODEL,
        device=device,
        lora_path=LORA_PATH,
        lora_scale=LORA_SCALE
    )

    generate_images(
        pipe=pipe,
        prompts=PROMPTS,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()