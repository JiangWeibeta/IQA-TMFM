import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import csv


def test_one_epoch(backbone, discriminator, save_output, tb_logger, test_dataloader,
                   logger_val, device, loss_type, criterion, current_epoch):

    discriminator.eval()

    avg_loss = 0
    accuracy = 0
    sum = 0

    with torch.no_grad():

        for data in tqdm(test_dataloader):

            sum += 1
            img_r = data["img_r"].to(device)
            img_a = data["img_a"].to(device)
            img_b = data["img_b"].to(device)
            label = data["label"].to(device)

            _, _, height, width = img_r.shape

            # batch_size * 4
            img_r = img_r.reshape(-1, 3, height, width)
            img_a = img_a.reshape(-1, 3, height, width)
            img_b = img_b.reshape(-1, 3, height, width)

            # batch_size * 3 * 3
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

            pred_logits, pred = discriminator(feature2x, feature3x)

            pred_logits = pred_logits.reshape(-1, 1)
            pred = pred.reshape(-1, 1)

            if loss_type == 'mse':
                loss = criterion(pred_logits, label)
            else:
                loss = F.binary_cross_entropy_with_logits(pred_logits, label.float())

            tmp = torch.where(pred > 0.5, 1, 0)
            avg_loss += loss
            accuracy += torch.sum(torch.eq(tmp, label)).item()

    logger_val.info(
        f"Test epoch {current_epoch}: ["
        f"loss: [{avg_loss / sum:.3f}] | "
        f"Accuracy: [{accuracy / sum:.3f}] | "
    )
    return avg_loss / sum
