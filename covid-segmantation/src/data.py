import os
from typing import Tuple, Union, Optional, Sequence

import numpy
import numpy as np
import torch
from matplotlib import pyplot, pyplot as plt


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
    # train_images = np.concatenate((images_medseg[train_indexes], images_radiopedia))
    # train_masks = np.concatenate((masks_medseg_recover[train_indexes], masks_radiopedia_recover))
    train_images = images_medseg[train_indexes]
    train_masks = masks_medseg_recover[train_indexes]
    val_images = images_medseg[val_indexes]
    val_masks = masks_medseg_recover[val_indexes]

    # batch_size = len(val_masks)

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
