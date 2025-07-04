import torch
from torch import device, cuda, optim, autocast, save
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import MulticlassAccuracy
from torchgeo.models import FarSeg

from loader import train_dl, test_dl


def train_model(model, device_hw, epoch_num, weight_decay, lr):
    # Set the optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    loss_fn = CrossEntropyLoss()
    gradient_scaler = torch.amp.GradScaler()

    # begin training
    for epoch in range(epoch_num):
        print(f"Epoch: {epoch}\n-------")
        epoch_loss = 0
        model.train()

        for images, masks in train_dl:
            images = images.to(device=device_hw)
            masks = masks.to(device=device_hw)

            # skip chips all water
            if images.max() == 1 and images.min() == 1:
                continue

            with autocast(device_hw.type):
                mask_prediction = model(images)
                loss = loss_fn(mask_prediction, masks)


            optimizer.zero_grad()
            gradient_scaler.scale(loss).backward()
            gradient_scaler.step(optimizer)
            gradient_scaler.update()
            epoch_loss += loss.item()

        # average loss per batch per epoch
        epoch_loss /= len(train_dl)
        scheduler.step(epoch_loss)

        # Test Error
        test_iou, test_acc, test_loss = 0, 0, 0
        model.eval()
        with torch.inference_mode():
            for images, masks in test_dl:
                images = images.to(device=device_hw)
                masks = masks.to(device=device_hw)

                with autocast(device_hw.type):
                    test_pred = model(images)
                    loss = loss_fn(test_pred, masks)
                    test_loss += loss

                metric = MulticlassAccuracy(num_classes=classes, average='macro').to(device_hw)
                pred = torch.argmax(test_pred, dim=1)
                test_acc += metric(pred, masks)

        avg_acc = test_acc / len(test_dl)
        avg_loss = test_loss / len(test_dl)

        print(f"\nTrain loss: {epoch_loss:.5f} | Test loss: {avg_loss:.5f}, Test Accuracy: {avg_acc:.2f}\n")

    print("Training Complete!")
    state_dict = model.state_dict()
    save(state_dict, "/mnt/d/SARFloodModel.pth")
    print("Model Saved")


if __name__ == '__main__':
    print("Using PyTorch version: ", torch.__version__)

    device = device('cuda' if cuda.is_available() else 'cpu')
    # device = device('cpu') for debug
    print(f"Training on {device}")

    epochs = 5
    learning_rate = 0.0001
    decay = 0.01
    classes = 2
    bands = 3

    model = FarSeg(backbone='resnet18', classes=classes, backbone_pretrained=True).to(device)

    try:
        train_model(
            model=model,
            device_hw=device,
            epoch_num=epochs,
            weight_decay=decay,
            lr=learning_rate
        )

    except cuda.OutOfMemoryError:
        print("Out of memory!")
        cuda.empty_cache()
