import numpy as np
import pandas as pd
import scipy
import scipy.ndimage
import torch
from torch import nn
from torchvision import transforms as T

from data import visualize
from lightning_module import CovidSegmenter


def predict_single_image(model, image_np, device, val_augs):
    model.eval()

    image_aug = val_augs(image=image_np)['image']

    mean = [0.485]
    std = [0.229]
    transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image_t = transform(image_aug)

    image_t = image_t.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_t)
        output = nn.Softmax(dim=1)(output)
        output = output.permute(0, 2, 3, 1)

    return output.squeeze(0).cpu().numpy()


def run_test_predictions(checkpoint_callback, datamodule, device, target_size):
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
    val_augs = datamodule.val_augs

    image_batch = np.stack([val_augs(image=img)['image'] for img in test_images], axis=0)

    print(f"Test image batch shape (after augs): {image_batch.shape}")

    output = np.zeros((len(test_images), target_size, target_size, 4))
    for i in range(len(test_images)):
        output[i] = predict_single_image(best_model, image_batch[i], device, val_augs)

    print(f"Output prediction shape: {output.shape}")
    test_masks_prediction = output > 0.5
    visualize(image_batch, test_masks_prediction, num_samples=len(test_images))

    print("Resizing test predictions to original size...")
    test_masks_prediction_original_size = scipy.ndimage.zoom(test_masks_prediction[..., :-2], (1, 2, 2, 1), order=0)
    print(f"Resized predictions shape: {test_masks_prediction_original_size.shape}")

    print("Creating submission file (sub.csv)...")
    frame = pd.DataFrame(
        data=np.stack(
            (np.arange(len(test_masks_prediction_original_size.ravel())),
             test_masks_prediction_original_size.ravel().astype(int)),
            axis=-1
        ),
        columns=['Id', 'Predicted']
    ).set_index('Id')
    frame.to_csv('sub.csv')
    print("Submission file created successfully.")
