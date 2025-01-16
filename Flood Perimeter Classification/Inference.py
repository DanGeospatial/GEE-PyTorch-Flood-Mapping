import torch
from torch import device, cuda
import segmentation_models_pytorch as smp



if __name__ == '__main__':
    print("Using PyTorch version: ", torch.__version__)

    device = device('cuda' if cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    classes = 1

    model = smp.Unet(encoder_name="resnet18", in_channels=1, classes=classes, activation="softmax").to(device)
    model.load_state_dict(torch.load("/mnt/d/SAR_Water_v1.pth", weights_only=True, map_location=device))
    model.eval()
    with torch.inference_mode():


