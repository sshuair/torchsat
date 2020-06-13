import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchsat.transforms.transforms_cls as T_cls
from torchsat.datasets.folder import ImageFolder
from torchsat.models.utils import get_model


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    for batch_idx, (image, target) in enumerate(data_loader):
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if batch_idx%print_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(image), len(data_loader.dataset),
                    100. * batch_idx / len(data_loader), loss.item()))


def evaluate(model, criterion, data_loader, device):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for image, target in data_loader:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss += criterion(output, target).item()

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        loss /= len(data_loader.dataset)/data_loader.batch_size

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset)))


def load_data(traindir, valdir):
    train_transform = T_cls.Compose([
        # T_cls.RandomHorizontalFlip(),
        # T_cls.RandomVerticalFlip(),
        T_cls.ToTensor(),
        # T_cls.Normalize(),
    ])
    val_transform = T_cls.Compose([
        T_cls.ToTensor(),
        # T_cls.Normalize(),
    ])
    dataset_train = ImageFolder(traindir, train_transform)
    dataset_val = ImageFolder(valdir, val_transform)

    return dataset_train, dataset_val


def main(args):
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if args.device == 'cuda' else 'cpu')

    # dataset and dataloader
    dataset_train, dataset_val = load_data(args.train_path, args.val_path)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True)

    # model
    model = get_model(args.model, args.num_classes, pretrained=args.pretrained)
    model.to(device)
    # from torchvision.models import resnet34
    # model = resnet34(pretrained=True)
    # model.fc = torch.nn.Linear(model.fc.in_features, args.num_classes)
    # model.to(device)

    # resume from previous trained checkpoint
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))

    # loss
    criterion = nn.CrossEntropyLoss()

    # optim and lr scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-8)

    for epoch in range(args.epochs):
        train_one_epoch(model, criterion, optimizer, train_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        evaluate(model, criterion, val_loader, device)
        torch.save(model.state_dict(), os.path.join(args.ckp_dir, "cls_epoch_{}.pth".format(epoch)))


def parse_args():
    parser = argparse.ArgumentParser(description='TorchSat Classification Training')
    parser.add_argument('--train-path', help='train dataset path')
    parser.add_argument('--val-path', help='validate dataset path')
    parser.add_argument('--model', default="resnet34", help='')
    parser.add_argument('--pretrained', default=True)

    parser.add_argument('--resume',default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('--num-classes', default=3, type=int, help='num of classes')
    parser.add_argument('--in-channels', default=3, type=int, help='input image channels')

    parser.add_argument('--device', default='cpu')
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')

    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--ckp-dir', default='./', help='path to save checkpoint')



    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
