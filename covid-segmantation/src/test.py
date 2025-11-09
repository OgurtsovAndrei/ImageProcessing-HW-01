import os
from typing import Optional

import numpy as np
import pandas as pd
import scipy
import scipy.ndimage
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms as T
import albumentations  # TTA: Импортируем albumentations
import torchmetrics

from data import visualize
from lightning_module import CovidSegmenter


def predict_batch(model: nn.Module, image_batch_t: torch.Tensor) -> torch.Tensor:
    """
    TTA: Новая функция, которая просто прогоняет батч тензоров через модель.
    """
    model.eval()
    with torch.no_grad():
        output = model(image_batch_t)
        # Применяем Softmax ЗДЕСЬ, чтобы усреднять вероятности
        output = nn.Softmax(dim=1)(output)
    return output


def run_test_predictions(checkpoint_callback, datamodule, device, target_size, miou_val: Optional[float]):
    print("\nStarting test predictions...")

    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path or not os.path.exists(best_model_path):
        print("No best model checkpoint found. Skipping test predictions.")
        return

    print(f"Loading best model from: {best_model_path}")
    best_model = CovidSegmenter.load_from_checkpoint(best_model_path)
    best_model.to(device)
    best_model.eval()

    datamodule.setup('fit')
    test_images = datamodule.test_images
    val_augs = datamodule.val_augs  # Это наш ресайз

    # TTA: Определяем трансформацию (нормализацию)
    mean = [0.485]
    std = [0.229]
    transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    # TTA: Определяем аугментацию для флипа
    hflip_aug = albumentations.HorizontalFlip(p=1.0)

    print(f"Running predictions with TTA (Horizontal Flip) on {len(test_images)} images...")

    output = np.zeros((len(test_images), target_size, target_size, 4))

    for i in range(len(test_images)):
        img_np = test_images[i]  # Оригинальное numpy изображение

        # --- 1. Оригинальное изображение ---
        img_orig_aug = val_augs(image=img_np)['image']  # Применяем ресайз
        img_orig_t = transform(img_orig_aug).to(device)  # Конвертируем в тензор

        # --- 2. Отраженное изображение ---
        img_flipped_np = hflip_aug(image=img_np)['image']  # Отражаем numpy
        img_flipped_aug = val_augs(image=img_flipped_np)['image']  # Применяем ресайз
        img_flipped_t = transform(img_flipped_aug).to(device)  # Конвертируем в тензор

        # --- 3. Пакетное предсказание ---
        # Стэкаем оба изображения в один батч [2, 1, H, W]
        batch_t = torch.stack([img_orig_t, img_flipped_t])

        # Получаем предсказания [2, 4, H, W]
        preds_t = predict_batch(best_model, batch_t)

        # --- 4. Усреднение ---
        pred_orig_t = preds_t[0]  # [4, H, W]

        # Берем отраженное предсказание и отражаем его обратно по оси W (dim=2)
        pred_flipped_restored_t = torch.flip(preds_t[1], dims=[2])  # [4, H, W]

        # Усредняем
        avg_pred_t = (pred_orig_t + pred_flipped_restored_t) / 2.0

        # --- 5. Сохранение результата ---
        # Конвертируем в numpy [H, W, 4] для сохранения
        avg_pred_np = avg_pred_t.permute(1, 2, 0).cpu().numpy()
        output[i] = avg_pred_np

    print(f"Output prediction shape (after TTA): {output.shape}")
    test_masks_prediction = output > 0.5

    # Визуализируем TTA-результаты
    # Нам нужен image_batch для visualize, создадим его из оригинальных аугментированных картинок
    image_batch_orig = np.stack([val_augs(image=img)['image'] for img in test_images], axis=0)
    visualize(image_batch_orig, test_masks_prediction, num_samples=len(test_images))

    print("Resizing test predictions to original size...")
    # TTA: Убедимся, что обрезаем 4-й класс (если он есть), но код ноутбука не обрезал...
    # Судя по вашему коду, у вас 4 класса, и вы обрезали 2. Оставим эту логику.
    # test_masks_prediction_original_size = scipy.ndimage.zoom(test_masks_prediction, (1, 2, 2, 1), order=0)

    # Код из ноутбука (и ваш) обрезает последние 2 класса. Это странно, но оставим как есть.
    test_masks_prediction_original_size = scipy.ndimage.zoom(test_masks_prediction[..., :-2], (1, 2, 2, 1), order=0)
    print(f"Resized predictions shape: {test_masks_prediction_original_size.shape}")

    print("Creating submission file (sub.csv)...")
    frame = pd.DataFrame(
        data=np.stack(
            (np.arange(len(test_masks_prediction_original_size.ravel())),
             test_masks_prediction_original_size.ravel().astype(int)),
            axis=1
        ),
        columns=('Id', 'Predicted')
    )
    frame.to_csv('sub.csv', index=False)

    print("Submission file created.")

    if miou_val:
        # TTA: Добавляем TTA_ в имя файла, чтобы отличать логи
        log_filename = f"submission_miou_{miou_val:.4f}_TTA.csv"
        frame.to_csv(log_filename, index=False)
        print(f"Submission file also saved to {log_filename}")


def run_val_tta_evaluation(
    checkpoint_callback,
    datamodule,
    device: torch.device,
    num_classes: int = 4,
    visualize_samples: int = 8,
):
    """
    Run horizontal flip TTA on the validation set and report mIoU.

    This mirrors the TTA strategy used in run_test_predictions but operates on
    the validation dataloader (tensor batches), so no albumentations are applied here.
    """
    print("\nRunning TTA evaluation on validation set...")

    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path or not os.path.exists(best_model_path):
        print("No best model checkpoint found. Skipping TTA validation evaluation.")
        return None

    print(f"Loading best model from: {best_model_path}")
    model = CovidSegmenter.load_from_checkpoint(best_model_path)
    model.to(device)
    model.eval()

    # Ensure datamodule is set up and get validation loader
    datamodule.setup('fit')
    val_loader = datamodule.val_dataloader()

    # Use the same metric definition as in CovidSegmenter (macro mIoU)
    miou_metric = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes, average='macro').to(device)

    first_batch_images = None
    first_batch_masks = None
    first_batch_probs = None

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Original
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            # Flipped
            images_flipped = torch.flip(images, dims=[3])  # flip width dimension
            logits_flip = model(images_flipped)
            probs_flip = torch.softmax(logits_flip, dim=1)
            probs_flip_restored = torch.flip(probs_flip, dims=[3])

            # Average probs
            probs_avg = (probs + probs_flip_restored) / 2.0
            preds = torch.argmax(probs_avg, dim=1)

            miou_metric.update(preds, masks)

            # Keep first batch for visualization
            if first_batch_images is None:
                first_batch_images = images.detach().cpu()
                first_batch_masks = masks.detach().cpu()
                first_batch_probs = probs_avg.detach().cpu()

    tta_miou = miou_metric.compute().item()
    print(f"[Val TTA] mIoU (macro) = {tta_miou:.4f}")

    # Optional visualization for the first batch
    if first_batch_images is not None and visualize_samples > 0:
        try:
            b = min(visualize_samples, first_batch_images.shape[0])
            imgs_np = first_batch_images[:b].permute(0, 2, 3, 1).numpy()  # [B,H,W,1]
            probs_np = first_batch_probs[:b].permute(0, 2, 3, 1).numpy()  # [B,H,W,C]
            masks_oh = F.one_hot(first_batch_masks[:b].long(), num_classes=num_classes).numpy().astype(np.float32)
            visualize(imgs_np, masks_oh, probs_np, num_samples=b)
        except Exception as viz_err:
            print(f"[Val TTA] Visualization failed: {viz_err}")

    return tta_miou