import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def train_one_epoch(backbone, discriminator, optimizer, save_output, criterion, scheduler, tb_logger,
                    train_dataloader, logger_train, device, loss_type, current_epoch):
    discriminator.train()
    current_step = current_epoch * len(train_dataloader)

    for data in tqdm(train_dataloader):

        current_step += 1
        img_r = data["img_r"].to(device)
        img_a = data["img_a"].to(device)
        img_b = data["img_b"].to(device)
        label = data["label"].to(device)

        _, _, height, width = img_r.shape

        # batch_size * 4
        img_r = img_r.reshape(-1, 3, height, width)
        img_a = img_a.reshape(-1, 3, height, width)
        img_b = img_b.reshape(-1, 3, height, width)

        img_r = backbone(img_r)

        feature_r2x = save_output.outputs[3]  # [-1, 192, 28, 28]
        feature_r3x = save_output.outputs[5]  # [-1, 384, 14, 14]
        save_output.outputs.clear()

        img_a = backbone(img_a)
        feature_a2x = save_output.outputs[3]  # [-1, 192, 28, 28]
        feature_a3x = save_output.outputs[5]  # [-1, 384, 14, 14]
        save_output.outputs.clear()

        img_b = backbone(img_b)
        feature_b2x = save_output.outputs[3]  # [-1, 192, 28, 28]
        feature_b3x = save_output.outputs[5]  # [-1, 384, 14, 14]
        save_output.outputs.clear()

        feature2x = torch.cat([feature_r2x, feature_a2x, feature_b2x], dim=0)
        feature3x = torch.cat([feature_r3x, feature_a3x, feature_b3x], dim=0)

        # feature1x = feature1x.reshape(-1, 4, feature1x.shape[1], feature1x.shape[2], feature1x.shape[3])
        # feature2x = feature2x.reshape(-1, 4, feature2x.shape[1], feature2x.shape[2], feature2x.shape[3])
        # feature3x = feature3x.reshape(-1, 4, feature3x.shape[1], feature3x.shape[2], feature3x.shape[3])
        # feature4x = feature4x.reshape(-1, 4, feature4x.shape[1], feature4x.shape[2], feature4x.shape[3])

        # bacth_size * 4

        pred_logits, pred = discriminator(feature2x, feature3x)

        pred_logits = pred_logits.reshape(-1, 1)
        pred = pred.reshape(-1, 1)


        loss = F.binary_cross_entropy_with_logits(pred_logits, label.float())

        tmp = torch.where(pred > 0.5, 1, 0)

        acc = torch.sum(torch.eq(tmp, label)).item() / label.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        if current_step % 100 == 0:
            tb_logger.add_scalar('{}'.format('[train]: loss'), loss.item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: acc'), acc, current_step)
            logger_train.info(
                f"Train epoch {current_epoch}: ["
                f"{100. * current_step / len(train_dataloader):.0f}%] "
                f"loss: [{loss.item():.3f}] | "
                f"Accuracy: [{acc:.3f}] | "
            )
