import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):  #Here to define whether to use bilinear
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


        
class Generator(nn.Module):
    def __init__(self, bilinear=False):
        super(Generator, self).__init__()
        self.input_channels = 4
        self.output_channels = 1
        self.bilinear = bilinear

        self.inc = DoubleConv(self.input_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor)   #Put , bilinear after factor if you want to use bilinear mode
        self.up2 = Up(512, 256 // factor)
        self.up3 = Up(256, 128 // factor)
        self.up4 = Up(128, 64)
        self.outconv = OutConv(64, self.output_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output_layer = self.outconv(x)
        return output_layer



class Discriminator(nn.Module):
    def __init__(self, in_channels1, in_channels2):
        super(Discriminator, self).__init__()

        self.c0 = nn.Conv2d(in_channels1 + in_channels2, 64, 4, stride=2, padding=2)
        self.c1 = nn.Conv2d(64, 128, 4, stride=2, padding=2)
        self.c2 = nn.Conv2d(128, 256, 4, stride=2, padding=2)
        self.c3 = nn.Conv2d(256, 512, 4, stride=1, padding=2)
        self.c4 = nn.Conv2d(512, 1, 4, stride=1, padding=2)

        self.bnc1 = nn.BatchNorm2d(128)
        self.bnc2 = nn.BatchNorm2d(256)
        self.bnc3 = nn.BatchNorm2d(512)        

    def forward(self, x1, x2):
        h = self.c0(torch.cat((x1, x2),1))
        h = self.bnc1(self.c1(F.leaky_relu(h, negative_slope=0.2)))
        h = self.bnc2(self.c2(F.leaky_relu(h, negative_slope=0.2)))
        h = self.bnc3(self.c3(F.leaky_relu(h, negative_slope=0.2)))
        h = self.c4(F.leaky_relu(h, negative_slope=0.2))
        h = F.sigmoid(h)

        return h