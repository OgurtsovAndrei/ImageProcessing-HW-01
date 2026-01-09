import torch
from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm

# Конфигурация по умолчанию
DATA_DIR = "data/jb"
OUTPUT_DIR = "output_flux"
BASE_MODEL = "black-forest-labs/FLUX.1-dev"
RANK = 64
LORA_ALPHA = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
SAVE_STEPS = 100
IMAGE_SIZE = 512  # Flux поддерживает различные размеры, но для обучения используем 512


class FluxImageDataset(Dataset):
    """Датасет для обучения LoRA на Flux."""

    def __init__(self, data_dir, size=512):
        self.data_dir = Path(data_dir)
        self.image_paths = list(self.data_dir.glob("*.png")) + list(self.data_dir.glob("*.jpg"))
        self.size = size

        # Трансформации для изображений
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        print(f"📁 Найдено {len(self.image_paths)} изображений в {data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Генерируем prompt из имени файла
        prompt = img_path.stem.replace("_", " ").replace("-", " ")

        return {
            "image": image,
            "prompt": prompt
        }


def train_flux_lora(
        data_dir,
        output_dir,
        model_id="black-forest-labs/FLUX.1-dev",
        rank=8,
        lora_alpha=8,
        learning_rate=1e-4,
        num_epochs=100,
        batch_size=1,
        gradient_accumulation_steps=4,
        save_steps=50,
        image_size=512
):
    """
    Обучение LoRA адаптера для модели Flux.1.
    
    Args:
        data_dir: Директория с изображениями для обучения
        output_dir: Директория для сохранения чекпоинтов
        model_id: ID модели Flux на HuggingFace
        rank: Ранг LoRA (размерность адаптера)
        lora_alpha: Альфа параметр LoRA (масштабирование)
        learning_rate: Скорость обучения
        num_epochs: Количество эпох
        batch_size: Размер батча
        gradient_accumulation_steps: Шаги накопления градиентов
        save_steps: Частота сохранения чекпоинтов
        image_size: Размер изображений для обучения
    """

    # Определяем устройство
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Используется устройство: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # Загружаем модель Flux
    print(f"📦 Загрузка модели Flux: {model_id}")
    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )

    # Извлекаем компоненты
    transformer = pipe.transformer
    text_encoder = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    vae = pipe.vae
    scheduler = pipe.scheduler

    # Перемещаем на устройство
    transformer.to(device)
    text_encoder.to(device)
    text_encoder_2.to(device)
    vae.to(device)

    # Замораживаем все компоненты кроме transformer
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # Настраиваем LoRA для Flux transformer
    print(f"⚙️  Настройка LoRA с rank={rank}, alpha={lora_alpha}")

    # Flux использует DiT (Diffusion Transformer) архитектуру
    # Целевые модули для LoRA в Flux transformer
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",  # Attention layers
            "proj_in", "proj_out",  # Projection layers
            "ff.net.0.proj", "ff.net.2",  # Feed-forward layers
        ],
    )

    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    # Создаем датасет и dataloader
    dataset = FluxImageDataset(data_dir, size=image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False
    )

    # Оптимизатор
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8
    )

    # Scheduler для learning rate
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs * len(dataloader),
        eta_min=learning_rate * 0.1
    )

    print(f"🚀 Начало обучения на {num_epochs} эпох...")
    print(f"   Размер датасета: {len(dataset)}")
    print(f"   Батчей на эпоху: {len(dataloader)}")
    print(f"   Эффективный batch size: {batch_size * gradient_accumulation_steps}")

    global_step = 0
    best_loss = float('inf')

    for epoch in range(num_epochs):
        transformer.train()
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Эпоха {epoch + 1}/{num_epochs}")

        for step, batch in enumerate(progress_bar):
            images = batch["image"].to(device)
            prompts = batch["prompt"]

            # Кодируем изображения в latent space
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Генерируем шум
            noise = torch.randn_like(latents)

            # Случайные timesteps для flow matching
            timesteps = torch.rand(latents.shape[0], device=device)

            # Flow matching: интерполяция между шумом и данными
            noisy_latents = (1 - timesteps.view(-1, 1, 1, 1)) * latents + timesteps.view(-1, 1, 1, 1) * noise

            # Кодируем промпты с помощью обоих text encoders
            with torch.no_grad():
                # CLIP text encoder
                text_inputs = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]

                # T5 text encoder
                text_inputs_2 = tokenizer_2(
                    prompts,
                    padding="max_length",
                    max_length=tokenizer_2.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                text_embeddings_2 = text_encoder_2(text_inputs_2.input_ids.to(device))[0]

            # Предсказываем velocity (направление потока)
            model_pred = transformer(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=text_embeddings_2,
                pooled_projections=text_embeddings,
                return_dict=False
            )[0]

            # Flow matching loss: предсказываем разницу между данными и шумом
            target = latents - noise
            loss = torch.nn.functional.mse_loss(model_pred, target, reduction="mean")
            loss = loss / gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Обновление весов
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * gradient_accumulation_steps
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                "lr": f"{current_lr:.2e}"
            })

            # Сохранение чекпоинтов
            if global_step > 0 and global_step % save_steps == 0:
                save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                transformer.save_pretrained(save_path)
                print(f"\n💾 Сохранен чекпоинт: {save_path}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"📊 Эпоха {epoch + 1}/{num_epochs} - Средний Loss: {avg_loss:.4f}")

        # Сохраняем лучшую модель
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_save_path = os.path.join(output_dir, "best_lora")
            os.makedirs(best_save_path, exist_ok=True)
            transformer.save_pretrained(best_save_path)
            print(f"⭐ Сохранена лучшая модель с loss={avg_loss:.4f}")

    # Сохраняем финальную модель
    final_save_path = os.path.join(output_dir, "final_lora")
    os.makedirs(final_save_path, exist_ok=True)
    transformer.save_pretrained(final_save_path)
    print(f"\n✅ Обучение завершено! Финальные веса LoRA сохранены в {final_save_path}")
    print(f"   Лучшая модель сохранена в {os.path.join(output_dir, 'best_lora')}")

    return final_save_path


def main():
    train_flux_lora(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        model_id=BASE_MODEL,
        rank=RANK,
        lora_alpha=LORA_ALPHA,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        save_steps=SAVE_STEPS,
        image_size=IMAGE_SIZE
    )


if __name__ == "__main__":
    main()
