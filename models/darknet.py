import torch.nn as nn

class Residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, block):
        super().__init__()
        self.block = block
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.1)
        )

        self.residual = nn.Sequential(
            nn.Conv2d(out_channel, in_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.1)
        )
    def forward(self, x):
        x = self.conv2(x)
        residual = x
        for i in range(0, self.block):
            x = self.residual(x)
        x += residual
        return x

class Darknet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        )
        self.residual_block1 = Residual_block(32, 64, 1)
        self.residual_block2 = Residual_block(64, 128, 2)
        self.residual_block3 = Residual_block(128, 256, 8)
        self.residual_block4 = Residual_block(256, 512, 8)
        self.residual_block5 = Residual_block(512, 1024, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        output3 = self.residual_block3(x)
        output4 = self.residual_block4(output3)
        output5 = self.residual_block5(output4)
        return output3, output4, output5