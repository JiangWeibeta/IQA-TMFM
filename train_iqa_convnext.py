import os
import random
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import ImageFile, Image
from models.backbones.convnext import convnext_tiny, convnext_small, convnext_base
from models.discriminators.discriminator import Discriminator
from parsers import train_options
from utils import *


def main():
    torch.backends.cudnn.benchmark = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    args = train_options()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    if not os.path.exists(os.path.join('./results', args.experiment)):
        os.makedirs(os.path.join('./results', args.experiment))

    setup_logger('train', os.path.join('./results', args.experiment), 'train_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)
    setup_logger('val', os.path.join('./results', args.experiment), 'val_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)

    logger_train = logging.getLogger('train')
    logger_val = logging.getLogger('val')
    tb_logger = SummaryWriter(log_dir='./tb_logger/' + args.experiment)

    if not os.path.exists(os.path.join('./results', args.experiment, 'checkpoints')):
        os.makedirs(os.path.join('./results', args.experiment, 'checkpoints'))

    transform = transforms.Compose(
        [RandCrop(args.patch_size, args.crop_time), Normalize(0.5, 0.5), RandHorizontalFlip(), RandRotation(), ToTensor()]
    )
    full_dataset = IQADataset(
        db_path = args.dataset,
        csv_file_path = args.csv_file_path,
        transform = transform
    )
    train_dataset, test_dataset = random_split(full_dataset, [int(0.8 * len(full_dataset)), len(full_dataset) - int(0.8 * len(full_dataset))])
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

    if args.backbone == "convnext_tiny":
        backbone = convnext_tiny(pretrained=True)
    elif args.backbone == "convnext_small":
        backbone = convnext_small(pretrained=True)
    elif args.backbone == "convnext_base":
        backbone = convnext_base(pretrained=True)
    else:
        raise ValueError("Backbone not supported")

    backbone = backbone.to(device)
    backbone.eval()

    config1x, config2x, config3x, config4x = discriminator_config()
    discriminator = Discriminator(config1x, config2x, config3x, config4x).to(device)

    hook_handles = []
    save_output = SaveOutput()

    for layer in backbone.modules():
        if isinstance(layer, nn.Sequential):
            handle = layer.register_forward_hook(save_output)
            hook_handles.append(handle)
    if args.loss == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'mse':
        criterion = nn.MSELoss()
    else:
        raise ValueError("Loss not supported")

    best_loss = float('inf')
    optimizer = optim.Adam(discriminator.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4,8], gamma=0.1)
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        discriminator.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        checkpoint = None
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        logger_train.info(f"learning-rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            backbone=backbone, discriminator=discriminator, optimizer=optimizer, save_output=save_output,
            criterion=criterion, scheduler=scheduler, tb_logger=tb_logger, train_dataloader=train_dataloader,
            logger_train=logger_train, device=device, loss_type=args.loss, current_epoch=epoch
        )
        scheduler.step()

        loss = test_one_epoch(
            backbone=backbone, discriminator=discriminator, save_output=save_output, tb_logger=tb_logger,
            test_dataloader=test_dataloader, logger_val=logger_val, device=device, loss_type=args.loss,
            criterion=criterion, current_epoch=epoch
        )

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": discriminator.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "loss": loss
                },
                is_best,
                os.path.join('./results', args.experiment, 'checkpoints', "checkpoint_%03d.pth.tar" % epoch)
            )
        if is_best:
            logger_val.info(f"Best loss: {best_loss}")
            logger_val.info(f"Best checkpoint is saved")

if __name__ == '__main__':
    main()
