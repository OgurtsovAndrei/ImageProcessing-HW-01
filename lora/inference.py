import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import os
from pathlib import Path

LORA_PATH = "output/final_lora"
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "generated_images2"
PROMPTS = ["Telegram Computer Vision Group Logo"]
NUM_INFERENCE_STEPS = 100
GUIDANCE_SCALE = 7.5
NUM_IMAGES_PER_PROMPT = 1
USE_LORA = False


def load_lora_pipeline(base_model_id, lora_path, device):
    print(f"Loading base model: {base_model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,  # float32 for MPS
        safety_checker=None
    )

    print(f"Loading LoRA weights from: {lora_path}")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)

    pipe = pipe.to(device)

    print("Pipeline loaded successfully!")
    return pipe


def load_base_pipeline(base_model_id, device):
    print(f"Loading base model: {base_model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,  # float32 for MPS
        safety_checker=None
    )

    pipe = pipe.to(device)

    print("Pipeline loaded successfully!")
    return pipe


def generate_images(
        pipe,
        prompts,
        output_dir,
        width=512,
        height=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        num_images_per_prompt=1
):
    os.makedirs(output_dir, exist_ok=True)

    for i, prompt in enumerate(prompts):
        print(f"\nGenerating image {i + 1}/{len(prompts)}")
        print(f"Prompt: {prompt}")
        print(f"Size: {width}x{height}")

        with torch.no_grad():
            images = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt
            ).images

        for j, image in enumerate(images):
            filename = f"generated_{width}x{height}_prompt{i + 1}_{j + 1}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            print(f"Saved: {filepath}")

    print(f"\nAll images saved to: {output_dir}")


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if USE_LORA:
        print("Using LoRA model")
        pipe = load_lora_pipeline(BASE_MODEL, LORA_PATH, device)
    else:
        print("Using default base model (LoRA disabled)")
        pipe = load_base_pipeline(BASE_MODEL, device)

    print("\n" + "=" * 60)
    print("GENERATING 512x512 IMAGES")
    print("=" * 60)
    output_dir_512 = os.path.join(OUTPUT_DIR, "512x512")
    generate_images(
        pipe=pipe,
        prompts=PROMPTS,
        output_dir=output_dir_512,
        width=512,
        height=512,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        num_images_per_prompt=NUM_IMAGES_PER_PROMPT
    )

    print("\n" + "=" * 60)
    print("GENERATING 1000x240 IMAGES")
    print("=" * 60)
    output_dir_1000x240 = os.path.join(OUTPUT_DIR, "1000x240")
    generate_images(
        pipe=pipe,
        prompts=PROMPTS,
        output_dir=output_dir_1000x240,
        width=1000,
        height=240,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        num_images_per_prompt=NUM_IMAGES_PER_PROMPT
    )

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE!")
    print("=" * 60)
    print(f"Generated images saved in: {OUTPUT_DIR}")
    print(f"  - 512x512 images: {output_dir_512}")
    print(f"  - 1000x240 images: {output_dir_1000x240}")


if __name__ == "__main__":
    main()
