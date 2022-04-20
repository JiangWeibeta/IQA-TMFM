import torch
import torch.nn as nn
import torch.nn.functional as F

from models.discriminators.modules.iqt import IQARegression


class Discriminator(nn.Module):
    def __init__(self, config1x, config2x, config3x, config4x, ch=[96, 192, 384, 768]) -> None:
        super().__init__()

        self.config1x = config1x
        self.config2x = config2x
        self.config3x = config3x
        self.config4x = config4x

        # self.discrinimator1x = IQARegression(config=config1x, ch=ch[0])
        self.discrinimator2x = IQARegression(config=config2x, ch=ch[1])
        self.discrinimator3x = IQARegression(config=config3x, ch=ch[2])
        # self.discrinimator4x = IQARegression(config=config4x, ch=ch[3])

        # self.res_conv1x = nn.Conv2d(ch[0] * 2, ch[0], kernel_size=1, stride=1)
        self.res_conv2x = nn.Conv2d(ch[1] * 2, ch[1], kernel_size=1, stride=1)
        self.res_conv3x = nn.Conv2d(ch[2] * 2, ch[2], kernel_size=1, stride=1)
        # self.res_conv4x = nn.Conv2d(ch[3] * 2, ch[3], kernel_size=1, stride=1)

        self.act = nn.GELU()

    def forward(self, feature2x, feature3x):
        batch_size = feature2x.size(0) // 3

        device = feature2x.device

        enc_inputs2x = torch.ones(batch_size, self.config2x.n_enc_seq+1).to(device)
        dec_inputs2x = torch.ones(batch_size, self.config2x.n_enc_seq+1).to(device)
        enc_inputs3x = torch.ones(batch_size, self.config3x.n_enc_seq+1).to(device)
        dec_inputs3x = torch.ones(batch_size, self.config3x.n_enc_seq+1).to(device)

        img_r2x, img_a2x, img_b2x = feature2x.chunk(3, 0)
        img_r3x, img_a3x, img_b3x = feature3x.chunk(3, 0)

        res_a2x = self.act(self.res_conv2x(torch.cat([img_a2x, img_r2x], dim=1)))
        res_a3x = self.act(self.res_conv3x(torch.cat([img_a3x, img_r3x], dim=1)))

        res_b2x = self.act(self.res_conv2x(torch.cat([img_b2x, img_r2x], dim=1)))
        res_b3x = self.act(self.res_conv3x(torch.cat([img_b3x, img_r3x], dim=1)))

        img_2x = torch.cat([res_a2x, res_b2x], dim=1)
        img_3x = torch.cat([res_a3x, res_b3x], dim=1)

        pred_2x_logits = self.discrinimator2x(enc_inputs2x, img_2x, dec_inputs2x, img_r2x)
        pred_3x_logits = self.discrinimator3x(enc_inputs3x, img_3x, dec_inputs3x, img_r3x)

        pred_logits = (pred_2x_logits + pred_3x_logits) / 2

        pred_logits = pred_logits.reshape(-1, 4)
        pred_logits = torch.mean(pred_logits, dim=1)

        pred = torch.sigmoid(pred_logits)

        return pred_logits, pred


