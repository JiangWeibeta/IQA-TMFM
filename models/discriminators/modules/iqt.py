import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.discriminators.modules.transformer import *


class IQARegression(nn.Module):
    def __init__(self, config, ch):
        super().__init__()
        self.config = config
        self.ch = ch

        self.conv_enc = nn.Conv2d(in_channels=ch * 2, out_channels=config.d_hidn, kernel_size=1)
        self.conv_dec = nn.Conv2d(in_channels=ch, out_channels=config.d_hidn, kernel_size=1)

        self.transformer = Transformer(self.config)
        self.projection = nn.Sequential(
            nn.Linear(self.config.d_hidn, self.config.d_MLP_head, bias=False),
            nn.ReLU(),
            nn.Linear(self.config.d_MLP_head, self.config.n_output, bias=False)
        )

    def forward(self, enc_inputs, enc_inputs_embed, dec_inputs, dec_inputs_embed):
        # batch x (320*6) x 29 x 29 -> batch x 256 x 29 x 29
        enc_inputs_embed = self.conv_enc(enc_inputs_embed)
        dec_inputs_embed = self.conv_dec(dec_inputs_embed)
        # batch x 256 x 29 x 29 -> batch x 256 x (29*29)
        b, c, h, w = enc_inputs_embed.size()
        enc_inputs_embed = torch.reshape(enc_inputs_embed, (b, c, h*w))
        enc_inputs_embed = enc_inputs_embed.permute(0, 2, 1)
        # batch x 256 x (29*29) -> batch x (29*29) x 256
        dec_inputs_embed = torch.reshape(dec_inputs_embed, (b, c, h*w))
        dec_inputs_embed = dec_inputs_embed.permute(0, 2, 1)

        # (bs, n_dec_seq+1, d_hidn), [(bs, n_head, n_enc_seq+1, n_enc_seq+1)], [(bs, n_head, n_dec_seq+1, n_dec_seq+1)], [(bs, n_head, n_dec_seq+1, n_enc_seq+1)]
        dec_outputs, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs = self.transformer(enc_inputs, enc_inputs_embed, dec_inputs, dec_inputs_embed)

        # (bs, n_dec_seq+1, d_hidn) -> (bs, d_hidn)
        # dec_outputs, _ = torch.max(dec_outputs, dim=1)    # original transformer
        dec_outputs = dec_outputs[:, 0, :]                  # in the IQA paper
        # dec_outputs = torch.mean(dec_outputs, dim=1)      # general idea

        # (bs, n_output)
        pred = self.projection(dec_outputs)

        return pred
