import torch
import torch.nn as nn
from models.classification import resnet50


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


class FC_Siam_Diff(nn.Module):
    def __init__(self, num_classes=1, in_channels=3, pretrained=True):
        super(FC_Siam_Diff, self).__init__()
        self.encoder = resnet50(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)
        self.encoder1 = nn.Sequential(self.encoder.conv1,  self.encoder.bn1,  self.encoder.relu)
        self.encoder2 = nn.Sequential(self.encoder.maxpool, self.encoder.layer1)
        self.encoder3 = self.encoder.layer2
        self.encoder4 = self.encoder.layer3
        self.encoder5 = self.encoder.layer4

        self.center = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                    DecoderBlock(2048, 2048, 2048))

        self.decoder5 = DecoderBlock(2048*2, 1024+1024//2, 1024)
        self.decoder4 = DecoderBlock(1024*2, 512+512//2, 512)
        self.decoder3 = DecoderBlock(512*2, 256+256//2, 256)
        self.decoder2 = DecoderBlock(256*2, 128+128//2, 64)  # resnet50 layer1 is only 64 channels
        self.decoder1 = DecoderBlock(64*2, 64+64//2, 32)

        self.out = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(16, num_classes, kernel_size=3, padding=1))


    def forward(self, x_pre: torch.tensor, x_post: torch.tensor):
        # pre image
        pre_encoder1 = self.encoder1(x_pre)  #batchsize, 64, 256, 256
        pre_encoder2 = self.encoder2(pre_encoder1)  #batchsize, 256, 128, 128
        pre_encoder3 = self.encoder3(pre_encoder2)  #batchsize, 512, 64, 64
        pre_encoder4 = self.encoder4(pre_encoder3)  #batchsize, 1024, 32, 32
        pre_encoder5 = self.encoder5(pre_encoder4)  #batchsize, 2048, 16, 16

        # post image
        post_encoder1 = self.encoder1(x_post)  #batchsize, 64, 256, 256
        post_encoder2 = self.encoder2(post_encoder1)  #batchsize, 256, 128, 128
        post_encoder3 = self.encoder3(post_encoder2)  #batchsize, 512, 64, 64
        post_encoder4 = self.encoder4(post_encoder3)  #batchsize, 1024, 32, 32
        post_encoder5 = self.encoder5(post_encoder4)  #batchsize, 2048, 16, 16

        diff_encoder1 = torch.abs(pre_encoder1 - post_encoder1)
        diff_encoder2 = torch.abs(pre_encoder2 - post_encoder2)
        diff_encoder3 = torch.abs(pre_encoder3 - post_encoder3)
        diff_encoder4 = torch.abs(pre_encoder4 - post_encoder4)
        diff_encoder5 = torch.abs(pre_encoder5 - post_encoder5)


        center = self.center(diff_encoder5)

        decoder5 = self.decoder5(torch.cat([diff_encoder5, center], dim=1))
        decoder4 = self.decoder4(torch.cat([diff_encoder4, decoder5], dim=1))
        decoder3 = self.decoder3(torch.cat([diff_encoder3, decoder4], dim=1))
        decoder2 = self.decoder2(torch.cat([diff_encoder2, decoder3], dim=1))
        decoder1 = self.decoder1(torch.cat([diff_encoder1, decoder2], dim=1))
        out = self.out(decoder1)

        return out

if __name__ == '__main__':
    import torch

    # x = torch.randn(size=(1, 1024, 16, 16))
    # test decoder block
    # decoder = DecoderBlock(1024, 600, 512)
    # decoder(x)

    x_pre = torch.randn(size=(2, 3, 512, 512))
    x_post = torch.randn(size=(2, 3, 512, 512))
    model = FC_Siam_Diff()
    out = model(x_pre, x_post)
    print(out.shape)
