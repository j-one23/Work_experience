import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models  # ✅ models import 추가

# 1. TPU-Compatible 모델 정의 (Upsample 기반)
class UNetMobileNetV2_TPU(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).features

        self.enc1 = base_model[0:4]
        self.enc2 = base_model[4:7]
        self.enc3 = base_model[7:14]
        self.enc4 = base_model[14:]

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
        self.up5 = up_block(16, 8)
        self.final = nn.Conv2d(8, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d1 = self.up1(e4)
        d1 = torch.cat([d1, e3], dim=1)
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d3 = self.up3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        d4 = self.up4(d3)
        d5 = self.up5(d4)

        return self.final(d5)

# 2. 메인 함수
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetMobileNetV2_TPU().to(device)
    model.load_state_dict(torch.load("unet_tpu_ready.pt", map_location=device))
    model.eval()

    video_path = "test.mp4"
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f" 비디오 파일을 열 수 없습니다: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H, W = frame.shape[:2]

        img = cv2.resize(frame, (256, 256))
        inp = img.astype(np.float32) / 255.0
        inp = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = torch.sigmoid(model(inp))[0, 0].cpu().numpy()

        mask = (pred > 0.5).astype(np.uint8) * 255
        mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        red_overlay = np.zeros((H, W, 3), dtype=np.uint8)
        red_overlay[..., 2] = mask_resized

        overlay = cv2.addWeighted(frame, 0.7, red_overlay, 0.3, 0)

        cv2.imshow("Lane Segmentation - Video", overlay)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
