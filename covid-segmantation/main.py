from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import albumentations
import numpy as np  # linear algebra
import segmentation_models_pytorch as smp
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
from torch import nn
from torch.utils.data import DataLoader
import torch
import scipy
import pandas as pd
from dataset import Dataset
from train import *
from plot import *

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# typed global used in prepare_data
batch_size: int = 0


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    global filename
    for dirname, _, filenames in os.walk('data/covid-segmentation'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

    prefix = 'data/covid-segmentation'

    images_radiopedia = np.load(os.path.join(prefix, 'images_radiopedia.npy')).astype(np.float32)
    masks_radiopedia = np.load(os.path.join(prefix, 'masks_radiopedia.npy')).astype(np.int8)
    images_medseg = np.load(os.path.join(prefix, 'images_medseg.npy')).astype(np.float32)
    masks_medseg = np.load(os.path.join(prefix, 'masks_medseg.npy')).astype(np.int8)

    test_images_medseg = np.load(os.path.join(prefix, 'test_images_medseg.npy')).astype(np.float32)

    print(images_radiopedia.shape)
    print(masks_radiopedia.shape)
    print(images_medseg.shape)
    print(masks_medseg.shape)
    print(test_images_medseg.shape)

    return images_radiopedia, masks_radiopedia, images_medseg, masks_medseg, test_images_medseg


def visualize(
    image_batch: Union[np.ndarray, torch.Tensor],
    mask_batch: Optional[Union[np.ndarray, torch.Tensor]] = None,
    pred_batch: Optional[Union[np.ndarray, torch.Tensor]] = None,
    num_samples: int = 8,
    hot_encode: bool = True,
) -> None:
    num_classes = mask_batch.shape[-1] if mask_batch is not None else 0
    fix, ax = plt.subplots(num_classes + 1, num_samples, figsize=(num_samples * 2, (num_classes + 1) * 2))

    for i in range(num_samples):
        ax_image = ax[0, i] if num_classes > 0 else ax[i]
        if hot_encode:
            ax_image.imshow(image_batch[i, :, :, 0], cmap='Greys')
        else:
            ax_image.imshow(image_batch[i, :, :])
        ax_image.set_xticks([])
        ax_image.set_yticks([])

        if mask_batch is not None:
            for j in range(num_classes):
                if pred_batch is None:
                    mask_to_show = mask_batch[i, :, :, j]
                else:
                    mask_to_show = np.zeros(shape=(*mask_batch.shape[1:-1], 3))
                    mask_to_show[..., 0] = pred_batch[i, :, :, j] > 0.5
                    mask_to_show[..., 1] = mask_batch[i, :, :, j]
                ax[j + 1, i].imshow(mask_to_show, vmin=0, vmax=1)
                ax[j + 1, i].set_xticks([])
                ax[j + 1, i].set_yticks([])

    plt.tight_layout()
    plt.show()


def onehot_to_mask(mask: np.ndarray, palette: Sequence[Sequence[int]]) -> np.ndarray:
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    return x


def preprocess_images(
    images_arr: np.ndarray, mean_std: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, Tuple[float, float]]:
    images_arr[images_arr > 500] = 500
    images_arr[images_arr < -1500] = -1500
    min_perc, max_perc = np.percentile(images_arr, 5), np.percentile(images_arr, 95)
    images_arr_valid = images_arr[(images_arr > min_perc) & (images_arr < max_perc)]
    mean, std = (images_arr_valid.mean(), images_arr_valid.std()) if mean_std is None else mean_std
    images_arr = (images_arr - mean) / std
    print(f'mean {mean}, std {std}')
    return images_arr, (mean, std)


def plot_hists(images1: np.ndarray, images2: Optional[np.ndarray] = None) -> None:
    plt.hist(images1.ravel(), bins=100, density=True, color='b', alpha=1 if images2 is None else 0.5)
    if images2 is not None:
        plt.hist(images2.ravel(), bins=100, density=True, alpha=0.5, color='orange')
    plt.show()


def prepare_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    global batch_size
    images_radiopedia, masks_radiopedia, images_medseg, masks_medseg, test_images_medseg = load_data()
    visualize(images_radiopedia[30:], masks_radiopedia[30:])

    palette = [[0], [1], [2], [3]]
    masks_radiopedia_recover = onehot_to_mask(masks_radiopedia, palette).squeeze()  # shape = (H, W)
    masks_medseg_recover = onehot_to_mask(masks_medseg, palette).squeeze()  # shape = (H, W)

    print('Hot encoded mask size: ', masks_radiopedia.shape)
    print('Paletted mask size:', masks_medseg_recover.shape)
    visualize(masks_medseg_recover[30:], hot_encode=False)

    images_radiopedia, mean_std = preprocess_images(images_radiopedia)
    images_medseg, _ = preprocess_images(images_medseg, mean_std)
    test_images_medseg, _ = preprocess_images(test_images_medseg, mean_std)

    plot_hists(images_medseg, images_radiopedia)

    val_indexes, train_indexes = list(range(24)), list(range(24, 100))
    train_images = np.concatenate((images_medseg[train_indexes], images_radiopedia))
    train_masks = np.concatenate((masks_medseg_recover[train_indexes], masks_radiopedia_recover))
    val_images = images_medseg[val_indexes]
    val_masks = masks_medseg_recover[val_indexes]

    batch_size = len(val_masks)

    del masks_medseg_recover
    del masks_radiopedia_recover
    del images_radiopedia
    del masks_radiopedia
    del images_medseg
    del masks_medseg

    return train_images, train_masks, val_images, val_masks, test_images_medseg


def mask_to_onehot(mask: np.ndarray, palette: Sequence[Sequence[int]]) -> torch.Tensor:
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in palette:
        # print('colour',colour)
        equality = np.equal(mask, colour)
        # print('equality',equality)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return torch.from_numpy(semantic_map)


SOURCE_SIZE: int = 512
TARGET_SIZE: int = 256
max_lr: float = 1e-3
epoch: int = 20
weight_decay: float = 1e-4

if __name__ == '__main__':
    train_images, train_masks, val_images, val_masks, test_images_medseg = prepare_data()

    train_augs = albumentations.Compose([
        albumentations.Rotate(limit=360, p=0.9, border_mode=cv2.BORDER_REPLICATE),
        albumentations.RandomSizedCrop((int(SOURCE_SIZE * 0.75), SOURCE_SIZE),
                                       (TARGET_SIZE, TARGET_SIZE),
                                       interpolation=cv2.INTER_NEAREST),
        albumentations.HorizontalFlip(p=0.5),

    ])

    val_augs = albumentations.Compose([
        albumentations.Resize(TARGET_SIZE, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    ])

    train_dataset = Dataset(train_images, train_masks, train_augs)
    val_dataset = Dataset(val_images, val_masks, val_augs)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    i, train_data = next(enumerate(train_dataloader))

    palette = [[0], [1], [2], [3]]
    mask_hot_encoded = mask_to_onehot(torch.unsqueeze(train_data[1], -1).numpy(), palette)
    # visualize(torch.unsqueeze(torch.squeeze(train_data[0],1),-1),mask_hot_encoded)
    visualize(train_data[0].permute(0, 2, 3, 1), mask_hot_encoded)

    model = smp.Unet('efficientnet-b2', in_channels=1, encoder_weights='imagenet', classes=4, activation=None,
                     encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
                                                steps_per_epoch=len(train_dataloader))

    history = fit(epoch, model, train_dataloader, val_dataloader, criterion, optimizer, sched)

    torch.save(model, 'Unet-efficientnet.pt')

    plot_loss(history)
    plot_score(history)
    plot_acc(history)

    image, mask = next(iter(val_dataloader))
    pred_mask, score, output = predict_image_mask_miou(model, image, mask)
    semantic_map = mask_to_onehot(torch.unsqueeze(mask, -1).numpy(), palette)

    # yellow is TP, red is FP, green is FN
    visualize(image, semantic_map, pred_batch=output.cpu())

    mob_miou = miou_score(model, val_dataloader)
    print("mob_miou:", mob_miou)

    del train_images
    del train_masks

    # test predictions

    image_batch = np.stack([val_augs(image=img)['image'] for img in test_images_medseg], axis=0)
    print(torch.from_numpy(image_batch).shape)
    print(image_batch[i].shape)
    # output = test_predict(model, torch.from_numpy(image_batch).permute(0, 3, 1,2))
    output = np.zeros((10, 256, 256, 4))
    for i in range(10):
        output[i] = test_predict(model, image_batch[i])
    print(output.shape)
    test_masks_prediction = output > 0.5
    visualize(image_batch, test_masks_prediction, num_samples=len(test_images_medseg))

    test_masks_prediction_original_size = scipy.ndimage.zoom(test_masks_prediction[..., :-2], (1, 2, 2, 1), order=0)
    print(test_masks_prediction_original_size.shape)

    frame = pd.DataFrame(
        data=np.stack(
            (np.arange(len(test_masks_prediction_original_size.ravel())),
             test_masks_prediction_original_size.ravel().astype(int)),
            axis=-1
        ),
        columns=['Id', 'Predicted']
    ) .set_index('Id')
    frame.to_csv('sub.csv')

