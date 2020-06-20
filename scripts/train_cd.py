import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ignite.metrics import IoU, Precision, Recall

import torchsat.transforms.transforms_cd as T
from torchsat.datasets.folder import ChangeDetectionDataset
from torchsat.models import FC_EF, FC_Siam_Conc, FC_Siam_Diff

def train_one_epoch(epoch, dataloader, model, criterion, optimizer, device, writer):
    print('train epoch {}'.format(epoch))
    model.train()
    for idx, (pre_img, post_img, targets) in enumerate(dataloader):
        pre_img, post_img, targets = pre_img.to(device), post_img.to(device), targets.to(device)
        outputs = model(pre_img, post_img)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('train-epoch:{} [{}/{}], loss: {:5.3}'.format(epoch, idx+1, len(dataloader), loss.item()))
        writer.add_scalar('train/loss', loss.item(), len(dataloader)*epoch+idx)


def evalidation(epoch, dataloader, model, criterion, device, writer, tb_test_imgs):
    print('\neval epoch {}'.format(epoch))
    model.eval()
    recall = Recall(lambda x: (x[0], x[1]))
    precision = Precision(lambda x: (x[0], x[1]))
    mean_recall = []
    mean_precision = []
    mean_loss = []
    with torch.no_grad():
        for idx, (pre_img, post_img, targets) in enumerate(dataloader):
            pre_img, post_img, targets = pre_img.to(device), post_img.to(device), targets.to(device)
            outputs = model(pre_img, post_img)
            loss = criterion(outputs, targets)

            preds = outputs.argmax(1)

            precision.update((preds, targets))
            recall.update((preds, targets))
            mean_loss.append(loss.item())
            mean_recall.append(recall.compute().item())
            mean_precision.append(precision.compute().item())

            # print('val-epoch:{} [{}/{}], loss: {:5.3}'.format(epoch, idx + 1, len(dataloader), loss.item()))
            writer.add_scalar('test/loss', loss.item(), len(dataloader) * epoch + idx)
            if idx < tb_test_imgs:
                writer.add_image('test/pre', pre_img[0], idx)
                writer.add_image('test/post', post_img[0], idx)
                writer.add_image('test/label', label[0], idx)
                writer.add_image('test/pred', preds, idx)

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
    train_transform = T.Compose([
        T.RandomCrop(512),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize(),
    ])
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(),
    ])
    dataset_train = ChangeDetectionDataset(traindir, extentions=kwargs['extensions'], transforms=train_transform, )
    dataset_val = ChangeDetectionDataset(valdir, extentions=kwargs['extensions'], transforms=val_transform)

    return dataset_train, dataset_val


def main(args):
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if args.device == 'cuda' else 'cpu')

    # dataset and dataloader
    train_data, val_data = load_data(args.train_path, args.val_path, extensions=args.extensions)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    # model
    # model = get_model(args.model, args.num_classes, pretrained=args.pretrained)
    # model = FC_EF(num_classes=args.num_classes)
    model = FC_Siam_Diff(num_classes=args.num_classes)
    model.to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        # TODO: resume learning rate

    # loss
    criterion = nn.CrossEntropyLoss().to(device)
    criterion = nn.BCELoss()

    # optim and lr scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-8)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    writer = SummaryWriter(args.ckp_dir)
    for epoch in range(args.epochs):
        writer.add_scalar('train/lr', lr_scheduler.get_lr()[0], epoch)
        train_one_epoch(epoch, train_loader, model, criterion, optimizer, device, writer)
        evalidation(epoch, val_loader, model, criterion, device, writer, args.tb_test_imgs)
        lr_scheduler.step()
        if epoch % 2 == 0:
            torch.save(model.state_dict(), os.path.join(args.ckp_dir, 'cd_epoch_{}.pth'.format(epoch)))


def parse_args():
    parser = argparse.ArgumentParser(description='TorchSat Change Detection Training Script')
    parser.add_argument('--train-path', help='train dataset path')
    parser.add_argument('--val-path', help='validate dataset path')
    parser.add_argument('--extensions', nargs='+', default='jpg', help='the train image extension')
    parser.add_argument('--model', default="unet34", help='model name. default, unet34')
    parser.add_argument('--pretrained', default=True, help='use ImageNet pretrained params')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--num-classes', default=3, type=int, help='num of classes')
    parser.add_argument('--in-channels', default=3, type=int, help='input image channels')

    parser.add_argument('--device', default='cpu', help='device')
    parser.add_argument('-b', '--batch-size', default=16, type=int, help='batch size')
    parser.add_argument('--epochs', default=90, type=int, help='epochs')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')

    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--ckp-dir', default='./', help='path to save checkpoint')
    parser.add_argument('--tb-test-imgs', default=10, help='the num of test image show in tensorboard')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
