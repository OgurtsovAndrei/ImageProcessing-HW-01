import os
import glob

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from freeze_utils import FreezeStrategy
from lightning_datamodule import CovidDataModule
from lightning_module import CovidSegmenter
from plot import generate_plots_from_logs
from test import run_test_predictions, run_val_tta_evaluation


# Те же базовые гиперпараметры, что и в main.py
SOURCE_SIZE: int = 512
TARGET_SIZE: int = SOURCE_SIZE
MAX_LR: float = 5e-5
WEIGHT_DECAY: float = 1e-5
L1_REG: float = 1e-6
BATCH_SIZE: int = 16


def find_latest_checkpoint(checkpoints_dir: str = "checkpoints") -> str:
    pattern = os.path.join(checkpoints_dir, "*.ckpt")
    candidates = glob.glob(pattern)
    if not candidates:
        return ""
    # Выбираем по времени модификации
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def read_epoch_from_ckpt(ckpt_path: str) -> int:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        return int(ckpt.get("epoch", 0))
    except Exception:
        return 0


def pick_accelerator() -> str:
    # В main.py жёстко задан 'mps'. Здесь добавим безопасный фолбэк.
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    torch.set_float32_matmul_precision('high')

    accelerator = pick_accelerator()
    device = torch.device("mps" if accelerator == "mps" else ("cuda" if accelerator == "gpu" else "cpu"))
    print(f"Активатор: {accelerator}. Устройство: {device}.")

    ckpt_path = CKPT_PATH.strip()
    if not ckpt_path:
        ckpt_path = find_latest_checkpoint("checkpoints")
        if ckpt_path:
            print(f"Не указан --ckpt, возьмём последний чекпоинт: {ckpt_path}")
    if not ckpt_path or not os.path.exists(ckpt_path):
        print("[ОШИБКА] Чекпоинт не найден. Укажите путь в CKPT_PATH или поместите .ckpt в папку 'checkpoints/'.")
        return

    saved_epoch = read_epoch_from_ckpt(ckpt_path)
    max_epochs = saved_epoch + int(ADD_EPOCHS)
    print(f"Продолжаем с эпохи {saved_epoch} ещё на {ADD_EPOCHS} эпох (max_epochs={max_epochs}).")

    datamodule = CovidDataModule(
        batch_size=BATCH_SIZE,
        source_size=SOURCE_SIZE,
        target_size=TARGET_SIZE,
        use_radiopedia=USE_RADIOPEDIA
    )

    model = CovidSegmenter(
        num_classes=4,
        max_lr=MAX_LR,
        weight_decay=WEIGHT_DECAY,
        freeze_strategy=FreezeStrategy.NO,
        l1_lambda=L1_REG,
    )

    # 4) Коллбэки/логгер — те же, что в main.py
    checkpoint_callback = ModelCheckpoint(
        monitor='val_miou',
        dirpath='checkpoints',
        filename='best_model-{epoch:02d}-{val_miou:.3f}',
        save_top_k=1,
        mode='max',
    )
    csv_logger = CSVLogger(save_dir="logs/")

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        logger=csv_logger,
        enable_progress_bar=True,
    )

    # 5) РЕЗЮМЕ ОБУЧЕНИЯ c восстановлением оптимизатора/шедулера/счётчиков
    print("Запускаем дообучение из чекпоинта с сохранением всех состояний оптимизатора...")
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    print("Дообучение завершено.")
    print(f"Лучший чекпоинт: {checkpoint_callback.best_model_path}")

    # 6) Пост-обработка: графики, валидационный TTA и предсказания на тесте — как в main.py
    log_dir = csv_logger.experiment.log_dir
    if log_dir:
        try:
            generate_plots_from_logs(log_dir)
        except Exception as e:
            print(f"[WARN] Не удалось сгенерировать графики: {e}")
    else:
        print("Не удалось определить папку логов — пропускаем построение графиков.")

    # Валидационный TTA (для сравнения с метриками эпох)
    try:
        tta_val_miou = run_val_tta_evaluation(checkpoint_callback, datamodule, device, num_classes=4, visualize_samples=8)
        if tta_val_miou is not None:
            print(f"Validation TTA mIoU (macro): {tta_val_miou:.4f}")
    except Exception as e:
        print(f"[WARN] Ошибка при TTA-оценке на валидации: {e}")

    # Сабмит с TTA на тесте
    best_score = checkpoint_callback.best_model_score
    miou_for_name = best_score.item() if best_score is not None else None
    run_test_predictions(checkpoint_callback, datamodule, device, TARGET_SIZE, miou_for_name)

CKPT_PATH: str = "checkpoints/best_model-epoch=69-val_miou=0.636.ckpt"  # путь к .ckpt; если пусто — возьмём самый свежий из папки checkpoints/
ADD_EPOCHS: int = 100
USE_RADIOPEDIA: bool = False


if __name__ == '__main__':
    main()
