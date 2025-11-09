from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import time

import numpy as np
from torch import nn
from tqdm import tqdm
import torch
from metrics import *
from torchvision import transforms as T
from torch.utils.data import DataLoader

device = torch.device("mps")

def get_lr(optimizer: torch.optim.Optimizer) -> float:
    for param_group in optimizer.param_groups:
        return float(param_group['lr'])


def fit(
    epochs: int,
    model: nn.Module,
    train_loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    patch: bool = False,
) -> Dict[str, List[float]]:
    # torch.cuda.empty_cache()
    train_losses: List[float] = []
    test_losses: List[float] = []
    val_iou: List[float] = []
    val_acc: List[float] = []
    train_iou: List[float] = []
    train_acc: List[float] = []
    lrs: List[float] = []
    min_loss = np.inf
    decrease = 1
    not_improve = 0

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss: float = 0.0
        iou_score: float = 0.0
        accuracy: float = 0.0
        # training loop
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            # training phase
            image_tiles, mask_tiles = data

            image = image_tiles.to(device)
            mask = mask_tiles.to(device)
            # forward
            output = model(image)

            loss = criterion(output, mask)
            # evaluation metrics
            iou_score += mIoU(output, mask)
            accuracy += pixel_accuracy(output, mask)
            # backward
            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()  # reset gradient

            # step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step()

            running_loss += loss.item()


        else:
            model.eval()
            test_loss: float = 0.0
            test_accuracy: float = 0.0
            val_iou_score: float = 0.0
            # validation loop
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    image_tiles, mask_tiles = data

                    image = image_tiles.to(device)
                    mask = mask_tiles.to(device)
                    output = model(image)
                    # evaluation metrics
                    val_iou_score += mIoU(output, mask)
                    test_accuracy += pixel_accuracy(output, mask)
                    # loss
                    loss = criterion(output, mask)
                    test_loss += loss.item()

            # calculation mean for each batch
            train_len = max(1, len(train_loader))
            val_len = max(1, len(val_loader))
            train_losses.append(running_loss / train_len)
            test_losses.append(test_loss / val_len)

            if min_loss > (test_loss / val_len):
                print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (test_loss / val_len)))
                min_loss = (test_loss / val_len)
                decrease += 1
                if decrease % 5 == 0:
                    print('saving model...')
                    torch.save(model, 'Unet_efficientnet_b2_mIoU-{:.3f}.pt'.format(val_iou_score / val_len))

            if (test_loss / val_len) > min_loss:
                not_improve += 1
                min_loss = (test_loss / val_len)
                print(f'Loss Not Decrease for {not_improve} time')
                if not_improve == 7:
                    print('Loss not decrease for 7 times, Stop Training')
                    break

            # iou
            val_iou.append(val_iou_score / val_len)
            train_iou.append(iou_score / train_len)
            train_acc.append(accuracy / train_len)
            val_acc.append(test_accuracy / val_len)
            print("Epoch:{}/{}..".format(e + 1, epochs),
                  "Train Loss: {:.3f}..".format(running_loss / train_len),
                  "Val Loss: {:.3f}..".format(test_loss / val_len),
                  "Train mIoU:{:.3f}..".format(iou_score / train_len),
                  "Val mIoU: {:.3f}..".format(val_iou_score / val_len),
                  "Train Acc:{:.3f}..".format(accuracy / train_len),
                  "Val Acc:{:.3f}..".format(test_accuracy / val_len),
                  "Time: {:.2f}m".format((time.time() - since) / 60))

    history: Dict[str, List[float]] = {'train_loss': train_losses, 'val_loss': test_losses,
               'train_miou': train_iou, 'val_miou': val_iou,
               'train_acc': train_acc, 'val_acc': val_acc,
               'lrs': lrs}
    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))
    return history


def predict_image_mask_miou(
    model: nn.Module,
    image: torch.Tensor,
    mask: torch.Tensor,
    mean: Sequence[float] = (0.485,),
    std: Sequence[float] = (0.229,),
) -> Tuple[torch.Tensor, float, torch.Tensor]:
    model.eval()
    # t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    # image = t(image)
    model.to(device)
    image = image.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        # image = image.unsqueeze(0)
        # mask = mask.unsqueeze(0)

        output = model(image)
        a, b, c, d = output.shape  # noqa: F841 - shapes unused
        score = mIoU(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, score, output.permute(0, 2, 3, 1)


def predict_image_mask_pixel(
    model: nn.Module,
    image: torch.Tensor,
    mask: torch.Tensor,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> Tuple[torch.Tensor, float]:
    model.eval()
    model.to(device)
    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        output = model(image)
        acc = pixel_accuracy(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, acc


def mask_to_onehot(mask: np.ndarray, palette: Sequence[Sequence[int]]) -> torch.Tensor:
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map: List[np.ndarray] = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return torch.from_numpy(semantic_map)

def miou_score(model: nn.Module, test_set: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[float]:
    score_iou: List[float] = []
    for i, data in enumerate(tqdm(test_set)):
        img, mask = data
        pred_mask, score, output = predict_image_mask_miou(model, img, mask)
        score_iou.append(score)
    return score_iou


def test_predict(
    model: nn.Module,
    image: np.ndarray | torch.Tensor,
    mean: Sequence[float] = (0.485,),
    std: Sequence[float] = (0.229,),
) -> np.ndarray:
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    if isinstance(image, np.ndarray):
        image_t = t(image)
    else:
        image_t = image  # assume already tensor

    model.eval()

    model.to(device)
    image_t = image_t.to(device)

    with torch.no_grad():
        output = model(torch.unsqueeze(image_t, 1))
        output = nn.Softmax(dim=1)(output)
        output = output.permute(0, 2, 3, 1)
    # return (H, W, C) ndarray for easier downstream numpy usage
    return output.squeeze(0).detach().cpu().numpy()