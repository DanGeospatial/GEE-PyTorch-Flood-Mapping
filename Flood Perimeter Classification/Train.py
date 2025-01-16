import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from torch import device, cuda, optim, autocast, save




# Ensure that image dimensions are correct
assert image.ndim == 4  # [batch_size, channels, H, W]






if __name__ == '__main__':
    print("Using PyTorch version: ", torch.__version__)

    device = device('cuda' if cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    network = deeplab(num_classes, num_bands, device)

    try:
        train_model(
            model=network,
            device_hw=device,
            epoch_num=epochs,
            lr=learning_rate
        )
    except cuda.OutOfMemoryError:
        print("Out of memory!")
        cuda.empty_cache()

