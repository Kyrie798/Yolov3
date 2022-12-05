import torch.nn as nn
from darknet import darknet53
import torch

def Conv(in_channel, out_chanel, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_chanel, kernel_size=kernel_size, stride=1, padding=(
            kernel_size - 1) // 2 if kernel_size else 0, bias=False),
        nn.BatchNorm2d(out_chanel),
        nn.LeakyReLU(0.1)
    )

def last_layer(in_channel, out_chanel, concat):
    return nn.Sequential(
            Conv(in_channel+concat, out_chanel, 1),
            Conv(out_chanel, in_channel, 3),
            Conv(in_channel, out_chanel, 1),
            Conv(out_chanel, in_channel, 3),
            Conv(in_channel, out_chanel, 1),
            Conv(out_chanel, in_channel, 3),
            nn.Conv2d(in_channel, 75, kernel_size=1,
                      stride=1, padding=0, bias=True)
        )

class Yolo3(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = darknet53()

        self.last_layer0 = last_layer(1024, 512, 0)

        self.last_layer1_conv = Conv(512, 256, 1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1 = last_layer(512, 256, 256)

        self.last_layer2_conv = Conv(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = last_layer(256, 128, 128)

    def forward(self, x):
        x2, x1, x0 = self.backbone(x)

        # 小尺寸输出13*13
        out0_branch = self.last_layer0[:5](x0)
        out0 = self.last_layer0[5:](out0_branch)

        # 中尺寸输出26*26
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1_branch = self.last_layer1[:5](x1_in)
        out1 = self.last_layer1[5:](out1_branch)

        # 大尺寸输出52*52
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2 = self.last_layer2(x2_in)

        return out0, out1, out2
