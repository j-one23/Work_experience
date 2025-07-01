import os
import cv2
import numpy as np
from glob import glob
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# 1. Custom Dataset
class LaneSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        img = np.transpose(img, (2, 0, 1))  # (3, 256, 256)
        mask = np.expand_dims(mask, axis=0)  # (1, 256, 256)

        return torch.from_numpy(img), torch.from_numpy(mask)

# 2. TPU-Compatible UNet + MobileNetV2
class UNetMobileNetV2_TPU(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).features

        self.enc1 = base_model[0:4]    # 24 channels
        self.enc2 = base_model[4:7]    # 32 channels
        self.enc3 = base_model[7:14]   # 96 channels
        self.enc4 = base_model[14:]    # 1280 channels

        def up_block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.up1 = up_block(1280, 96)
        self.up2 = up_block(96 + 96, 32)
        self.up3 = up_block(32 + 32, 24)
        self.up4 = up_block(24 + 24, 16)
        self.up5 = up_block(16, 8)  # Extra upsampling to reach 256x256

        self.final = nn.Conv2d(8, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d1 = self.up1(e4)         # -> 16x16
        d1 = torch.cat([d1, e3], dim=1)
        d2 = self.up2(d1)         # -> 32x32
        d2 = torch.cat([d2, e2], dim=1)
        d3 = self.up3(d2)         # -> 64x64
        d3 = torch.cat([d3, e1], dim=1)
        d4 = self.up4(d3)         # -> 128x128
        d5 = self.up5(d4)         # -> 256x256

        return self.final(d5)

# 3. Training function
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LaneSegmentationDataset("runs/bdd100k/images/train", "runs/bdd100k/seg_annotation/train")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # batch size 감소 시 메모리 이점

    model = UNetMobileNetV2_TPU().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()

        print(f"\n[Epoch {epoch+1}/{epochs}] ---------------------------")
        for i, (images, masks) in enumerate(dataloader, 1):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if i % 5 == 0 or i == len(dataloader):
                print(f"  Batch {i}/{len(dataloader)} - Loss: {loss.item():.4f}")

        duration = time.time() - start_time
        avg_loss = epoch_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1} completed in {duration:.2f}s - Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "unet_tpu_ready.pt")
    print("📦 TPU-ready 모델 저장 완료: unet_tpu_ready.pt")

if __name__ == "__main__":
    train()
