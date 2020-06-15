import torch
import torch.nn as nn
import torch.nn.functional as F
from ..classification import resnet


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(mid_channels),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True))
        # self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.up(x)
        return x


class UNet50(nn.Module):
    def __init__(self, num_classes=2, in_channels=3, pretrained=True):
        super(UNet50, self).__init__()
        from torchvision.models import resnet101
        # self.encoder = resnet50(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)
        self.encoder = resnet.resnet50(num_classes=num_classes, pretrained=pretrained)
        self.encoder1 = nn.Sequential(self.encoder.conv1,  self.encoder.bn1,  self.encoder.relu)
        self.encoder2 = nn.Sequential(self.encoder.maxpool, self.encoder.layer1)
        self.encoder3 = self.encoder.layer2
        self.encoder4 = self.encoder.layer3
        self.encoder5 = self.encoder.layer4

        self.center = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                    DecoderBlock(2048, 2048, 2048))

        self.decoder5 = DecoderBlock(4096, 1024+1024//2, 1024)
        self.decoder4 = DecoderBlock(2048, 512+512//2, 512)
        self.decoder3 = DecoderBlock(1024, 256+256//2, 256)
        self.decoder2 = DecoderBlock(512, 128+128//2, 64)
        self.decoder1 = DecoderBlock(128, 64+64//2, 32)

        self.out = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(16, num_classes, kernel_size=3, padding=1))


    # def forward(self, x_pre: torch.tensor, x_post: torch.tensor):
    def forward(self, x):
        # suppose x: (batch_size, 3, 512, 512)
        encoder1 = self.encoder1(x)  #batchsize, 64, 256, 256
        encoder2 = self.encoder2(encoder1)  #batchsize, 256, 128, 128
        encoder3 = self.encoder3(encoder2)  #batchsize, 512, 64, 64
        encoder4 = self.encoder4(encoder3)  #batchsize, 1024, 32, 32
        encoder5 = self.encoder5(encoder4)  #batchsize, 2048, 16, 16
        center = self.center(encoder5)
        decoder5 = self.decoder5(torch.cat([encoder5, center], dim=1))
        decoder4 = self.decoder4(torch.cat([encoder4, decoder5], dim=1))
        decoder3 = self.decoder3(torch.cat([encoder3, decoder4], dim=1))
        decoder2 = self.decoder2(torch.cat([encoder2, decoder3], dim=1))
        decoder1 = self.decoder1(torch.cat([encoder1, decoder2], dim=1))
        out = self.out(decoder1)

        return out

if __name__ == '__main__':
    import torch

    # x = torch.randn(size=(1, 1024, 16, 16))
    # test decoder block
    # decoder = DecoderBlock(1024, 600, 512)
    # decoder(x)

    x = torch.randn(size=(2, 3, 512, 512))
    model = FC_EF()
    out = model(x)
    print(out.shape)
