import torch
import torch.nn as nn
from torchvision import models

# ----------------------------
#  1.  (UNet + MobileNetV2)
# ----------------------------
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

        d1 = self.up1(e4)
        d1 = torch.cat([d1, e3], dim=1)
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d3 = self.up3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        d4 = self.up4(d3)
        d5 = self.up5(d4)

        return self.final(d5)

# ----------------------------
# ? 2. PyTorch to ONNX model
# ----------------------------
def convert_to_onnx():
    model = UNetMobileNetV2_TPU()
    model.load_state_dict(torch.load("unet_tpu_int8.pt", map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256)

    torch.onnx.export(
        model,
        dummy_input,
        "unet_tpu_int8.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=11
    )

    print(" PyTorch to ONNX 생성 파일명: unet_tpu_ready.onnx")

if __name__ == "__main__":
    convert_to_onnx()
