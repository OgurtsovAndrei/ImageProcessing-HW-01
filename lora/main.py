import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel, FluxPipeline
from transformers import CLIPTextModel
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm

DATA_DIR = "data/jb"
OUTPUT_DIR = "output"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
RANK = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
SAVE_STEPS = 50


class ImageDataset(Dataset):
    def __init__(self, data_dir, size=512):
        self.data_dir = Path(data_dir)
        self.image_paths = list(self.data_dir.glob("*.png")) + list(self.data_dir.glob("*.jpg"))
        self.size = size

        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        print(f"Found {len(self.image_paths)} images in {data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        prompt = img_path.stem.replace("_", " ").replace("-", " ")

        return {
            "image": image,
            "prompt": prompt
        }


def train_lora(
        data_dir,
        output_dir,
        model_id="runwayml/stable-diffusion-v1-5",
        rank=4,
        learning_rate=1e-4,
        num_epochs=100,
        batch_size=1,
        gradient_accumulation_steps=4,
        save_steps=50
):
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading Stable Diffusion model: {model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None
    )

    unet = pipe.unet
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    unet.to(device)
    text_encoder.to(device)
    vae.to(device)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    print(f"Configuring LoRA with rank={rank}")
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    dataset = ImageDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

    print(f"Starting training for {num_epochs} epochs...")
    global_step = 0

    for epoch in range(num_epochs):
        unet.train()
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for step, batch in enumerate(progress_bar):
            images = batch["image"].to(device)
            prompts = batch["prompt"]

            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.no_grad():
                text_inputs = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]

            noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample

            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            loss = loss / gradient_accumulation_steps

            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * gradient_accumulation_steps
            progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})

            if global_step > 0 and global_step % save_steps == 0:
                save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                unet.save_pretrained(save_path)
                print(f"Saved checkpoint to {save_path}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

    final_save_path = os.path.join(output_dir, "final_lora")
    os.makedirs(final_save_path, exist_ok=True)
    unet.save_pretrained(final_save_path)
    print(f"Training complete! Final LoRA weights saved to {final_save_path}")

    return final_save_path


if __name__ == "__main__":
    train_lora(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        model_id=MODEL_ID,
        rank=RANK,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        save_steps=SAVE_STEPS
    )
