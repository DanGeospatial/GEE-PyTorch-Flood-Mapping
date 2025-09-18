import torch
import torch.nn as nn
import torch.nn.functional as fn
from torchvision import models
from torchvision.models import ResNet18_Weights


# lightweight 2D UNet with skip connections
class UNet2D(nn.Module):
    def __init__(self, in_ch=3, pretrained=False, norm_layer=nn.BatchNorm2d):
        super().__init__()
        encoder = models.resnet18(
            weights=ResNet18_Weights.DEFAULT if pretrained else None
        )
        encoder.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.enc1 = nn.Sequential(*list(encoder.children())[:5])   # 64
        self.enc2 = encoder.layer1   # 64
        self.enc3 = encoder.layer2   # 128
        self.enc4 = encoder.layer3   # 256
        self.enc5 = encoder.layer4   # 512 (bottleneck)

        # Decoder with skips
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), norm_layer(256), nn.ReLU(inplace=True)
        )
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), norm_layer(128), nn.ReLU(inplace=True)
        )
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), norm_layer(64), nn.ReLU(inplace=True)
        )
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), norm_layer(64), nn.ReLU(inplace=True)
        )

    def forward_encoder(self, x):
        e1 = self.enc1(x)  # 64
        e2 = self.enc2(e1) # 64
        e3 = self.enc3(e2) # 128
        e4 = self.enc4(e3) # 256
        e5 = self.enc5(e4) # 512
        return e1, e2, e3, e4, e5

    def forward_decoder(self, e1, e2, e3, e4, bottleneck):
        d4 = self.up4(bottleneck)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = fn.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return d1


class UNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=2):
        super().__init__()
        self.unet = UNet2D(in_ch=in_ch)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        e1, e2, e3, e4, e5 = self.unet.forward_encoder(x)

        d1 = self.unet.forward_decoder(e1, e2, e3, e4, e5)
        out = self.final(d1)
        out = fn.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out