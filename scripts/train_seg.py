import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ignite.metrics import IoU, Precision, Recall

import torchsat.transforms.transforms_seg as T_seg
from torchsat.datasets.folder import SegmentationDataset
from torchsat.models.utils import get_model
from torchsat.models.segmentation import unet_v2

"""
from RoboSat
U-Net inspired encoder-decoder architecture with a ResNet encoder as proposed by Alexander Buslaev.
See:
- https://arxiv.org/abs/1505.04597 - U-Net: Convolutional Networks for Biomedical Image Segmentation
- https://arxiv.org/abs/1411.4038  - Fully Convolutional Networks for Semantic Segmentation
- https://arxiv.org/abs/1512.03385 - Deep Residual Learning for Image Recognition
- https://arxiv.org/abs/1801.05746 - TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation
- https://arxiv.org/abs/1806.00844 - TernausNetV2: Fully Convolutional Network for Instance Segmentation
"""

import torch
import torch.nn as nn

from torchvision.models import resnet50

from torchsat.models.segmentation.unet_v2 import UNet50

class ConvRelu(nn.Module):
    """3x3 convolution followed by ReLU activation building block.
    """

    def __init__(self, num_in, num_out):
        """Creates a `ConvReLU` building block.
        Args:
          num_in: number of input feature maps
          num_out: number of output feature maps
        """

        super().__init__()

        self.block = nn.Conv2d(num_in, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        """The networks forward pass for which autograd synthesizes the backwards pass.
        Args:
          x: the input tensor
        Returns:
          The networks output tensor.
        """

        return nn.functional.relu(self.block(x), inplace=True)


class DecoderBlock(nn.Module):
    """Decoder building block upsampling resolution by a factor of two.
    """

    def __init__(self, num_in, num_out):
        """Creates a `DecoderBlock` building block.
        Args:
          num_in: number of input feature maps
          num_out: number of output feature maps
        """

        super().__init__()

        self.block = ConvRelu(num_in, num_out)

    def forward(self, x):
        """The networks forward pass for which autograd synthesizes the backwards pass.
        Args:
          x: the input tensor
        Returns:
          The networks output tensor.
        """

        return self.block(nn.functional.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """The "U-Net" architecture for semantic segmentation, adapted by changing the encoder to a ResNet feature extractor.
       Also known as AlbuNet due to its inventor Alexander Buslaev.
    """

    def __init__(self, num_classes, num_channels=3, num_filters=32, pretrained=True):
        """Creates an `UNet` instance for semantic segmentation.
        Args:
          num_classes: number of classes to predict.
          pretrained: use ImageNet pre-trained backbone feature extractor
        """

        super().__init__()

        # Todo: make input channels configurable, not hard-coded to three channels for RGB
        self.resnet = resnet50(pretrained=True)
        # self.resnet = resnet50(num_classes, num_channels=num_channels, pretrained=pretrained) #6-channel

        # Access resnet directly in forward pass; do not store refs here due to
        # https://github.com/pytorch/pytorch/issues/8392

        self.center = DecoderBlock(2048, num_filters * 8)

        self.dec0 = DecoderBlock(2048 + num_filters * 8, num_filters * 8)
        self.dec1 = DecoderBlock(1024 + num_filters * 8, num_filters * 8)
        self.dec2 = DecoderBlock(512 + num_filters * 8, num_filters * 2)
        self.dec3 = DecoderBlock(256 + num_filters * 2, num_filters * 2 * 2)
        self.dec4 = DecoderBlock(num_filters * 2 * 2, num_filters)
        self.dec5 = ConvRelu(num_filters, num_filters)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        """The networks forward pass for which autograd synthesizes the backwards pass.
        Args:
          x: the input tensor
        Returns:
          The networks output tensor.
        """
        size = x.size()
        assert size[-1] % 32 == 0 and size[-2] % 32 == 0, "image resolution has to be divisible by 32 for resnet"

        enc0 = self.resnet.conv1(x)
        enc0 = self.resnet.bn1(enc0)
        enc0 = self.resnet.relu(enc0)
        enc0 = self.resnet.maxpool(enc0)

        enc1 = self.resnet.layer1(enc0)
        enc2 = self.resnet.layer2(enc1)
        enc3 = self.resnet.layer3(enc2)
        enc4 = self.resnet.layer4(enc3)

        center = self.center(nn.functional.max_pool2d(enc4, kernel_size=2, stride=2))

        dec0 = self.dec0(torch.cat([enc4, center], dim=1))
        dec1 = self.dec1(torch.cat([enc3, dec0], dim=1))
        dec2 = self.dec2(torch.cat([enc2, dec1], dim=1))
        dec3 = self.dec3(torch.cat([enc1, dec2], dim=1))
        dec4 = self.dec4(dec3)
        dec5 = self.dec5(dec4)

        return self.final(dec5)

def train_one_epoch(epoch, dataloader, model, criterion, optimizer, device, writer):
    print('train epoch {}'.format(epoch))
    model.train()
    for idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        # outputs = torch.squeeze(torch.sigmoid(model(inputs)))
        # loss = criterion(outputs, targets.type(torch.float))
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('train-epoch:{} [{}/{}], loss: {:5.3}'.format(epoch, idx+1, len(dataloader), loss.item()))
        writer.add_scalar('train/loss', loss.item(), len(dataloader)*epoch+idx)


def evalidation(epoch, dataloader, model, criterion, device, writer):
    print('\neval epoch {}'.format(epoch))
    model.eval()
    recall = Recall(lambda x: (x[0], x[1]))
    precision = Precision(lambda x: (x[0], x[1]))
    mean_recall = []
    mean_precision = []
    mean_loss = []
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # outputs = torch.squeeze(torch.sigmoid(model(inputs)))
            # loss = criterion(outputs, targets.type(torch.float))
            # preds = outputs > 0.5
            # preds = preds.type(torch.int8)

            preds = outputs.argmax(1)

            precision.update((torch.squeeze(preds), torch.squeeze(targets)))
            recall.update((torch.squeeze(preds), torch.squeeze(targets)))
            mean_loss.append(loss.item())
            mean_recall.append(recall.compute().item())
            mean_precision.append(precision.compute().item())

            # print('val-epoch:{} [{}/{}], loss: {:5.3}'.format(epoch, idx + 1, len(dataloader), loss.item()))
            writer.add_scalar('test/loss', loss.item(), len(dataloader) * epoch + idx)

    mean_precision, mean_recall = np.array(mean_precision).mean(), np.array(mean_recall).mean()
    f1 = mean_precision * mean_recall * 2 / (mean_precision + mean_recall + 1e-20)

    print('precision: {:07.5}, recall: {:07.5}, f1: {:07.5}\n'.format(mean_precision, mean_recall, f1))
    writer.add_scalar('test/epoch-loss', np.array(mean_loss).mean(), epoch)
    writer.add_scalar('test/f1', f1, epoch)
    writer.add_scalar('test/precision', mean_precision, epoch)
    writer.add_scalar('test/recall', mean_recall, epoch)


def load_data(traindir, valdir, **kwargs):
    """generate the train and val dataloader, you can change this for your specific task
    Args:
        traindir (str): train dataset dir
        valdir (str): validation dataset dir
    Returns:
        tuple: the train dataset and validation dataset
    """
    train_transform = T_seg.Compose([
        T_seg.RandomCrop(512),
        # T_seg.RandomHorizontalFlip(), #TODO: bug should be fix
        # T_seg.RandomVerticalFlip(),
        T_seg.ToTensor(),
        T_seg.Normalize(),
    ])
    val_transform = T_seg.Compose([
        T_seg.RandomCrop(512),
        T_seg.ToTensor(),
        T_seg.Normalize(),
    ])
    dataset_train = SegmentationDataset(traindir, extentions=kwargs['extensions'], transforms=train_transform, )
    dataset_val = SegmentationDataset(valdir, extentions=kwargs['extensions'], transforms=val_transform)

    return dataset_train, dataset_val


def main(args):
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if args.device == 'cuda' else 'cpu')

    # dataset and dataloader
    train_data, val_data = load_data(args.train_path, args.val_path, extensions=args.extensions)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    # model
    model = get_model(args.model, args.num_classes, pretrained=args.pretrained)
    # model = UNet(args.num_classes)
    model = UNet50(num_classes=args.num_classes)
    model.to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        # TODO: resume learning rate

    # loss
    criterion = nn.CrossEntropyLoss().to(device)
    # criterion = nn.BCELoss()

    # optim and lr scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-8)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)

    writer = SummaryWriter(args.ckp_dir)
    for epoch in range(args.epochs):
        writer.add_scalar('train/lr', lr_scheduler.get_lr()[0], epoch)
        train_one_epoch(epoch, train_loader, model, criterion, optimizer, device, writer)
        lr_scheduler.step()
        evalidation(epoch, val_loader, model, criterion, device, writer)
        torch.save(model.state_dict(), os.path.join(args.ckp_dir, 'seg_epoch_{}.pth'.format(epoch)))


def parse_args():
    parser = argparse.ArgumentParser(description='TorchSat Segmentation Training Script')
    parser.add_argument('--train-path', default='projects/seg/train', help='train dataset path')
    parser.add_argument('--val-path', default='projects/seg/train', help='validate dataset path')
    parser.add_argument('--extensions', nargs='+', default='jpg', help='the train image extension')
    parser.add_argument('--model', default="unet34", help='')
    parser.add_argument('--pretrained', default=True)

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--num-classes', default=2, type=int, help='num of classes')
    parser.add_argument('--in-channels', default=3, type=int, help='input image channels')

    parser.add_argument('--device', default='cuda')
    parser.add_argument('-b', '--batch-size', default=2, type=int)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')

    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--ckp-dir', default='./', help='path to save checkpoint')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
