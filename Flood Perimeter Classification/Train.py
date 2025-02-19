import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from torch import device, cuda, optim, autocast, save
from data.loader import train_dl, val_dl, test_dl


def train_model(model, device_hw, epoch_num, lr):

    # Set the optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    loss_fn = nn.BCEWithLogitsLoss()
    gradient_scaler = torch.amp.GradScaler()

    # begin training
    for epoch in range(epoch_num):
        print(f"Epoch: {epoch}\n-------")
        epoch_loss = 0
        model.train()

        for img, label in train_dl:
            images = img.to(device=device_hw)
            masks = label.to(device=device_hw)

            with autocast(device_hw.type):
                mask_prediction = model(images)
                loss = loss_fn(mask_prediction, masks.float())


            gradient_scaler.scale(loss).backward()
            gradient_scaler.step(optimizer)
            gradient_scaler.update()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        # average loss per batch per epoch
        epoch_loss /= len(train_dl)

        # Test Error
        test_loss = 0
        test_iou = []
        model.eval()
        with torch.inference_mode():
            for img, label in test_dl:
                images = img.to(device=device_hw)
                masks = label.to(device=device_hw)

                with autocast(device_hw.type):
                    test_pred = model(images)
                    test_loss += loss_fn(test_pred, masks.float())
                    tp, fp, fn, tn = smp.metrics.get_stats((test_pred.sigmoid() > 0.5).long(), masks.long(), mode='binary')
                    test_iou.append({"tp": tp, "fp": fp, "fn": fn, "tn": tn})

            test_loss /= len(test_dl)

            tp = torch.cat([x["tp"] for x in test_iou])
            fp = torch.cat([x["fp"] for x in test_iou])
            fn = torch.cat([x["fn"] for x in test_iou])
            tn = torch.cat([x["tn"] for x in test_iou])

            IoU = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise").item()

        scheduler.step(test_loss)

        print(f"\nTrain loss: {epoch_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {IoU:.2f}\n")

    print("Training Complete!")
    # state_dict = model.state_dict()
    # save(state_dict, "/mnt/d/SAR_Water_v1.pth")
    print("Model Saved")

if __name__ == '__main__':
    print("Using PyTorch version: ", torch.__version__)

    device = device('cuda' if cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    epochs = 10
    classes = 1
    learning_rate = 1e-5

    model = smp.Linknet(encoder_name="efficientnet-b2", in_channels=3, classes=classes, encoder_weights="advprop").to(device)

    try:
        train_model(
            model=model,
            device_hw=device,
            epoch_num=epochs,
            lr=learning_rate
        )
    except cuda.OutOfMemoryError:
        print("Out of memory!")
        cuda.empty_cache()

