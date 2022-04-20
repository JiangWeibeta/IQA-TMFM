import os
import json

class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.load(f.read())
            return Config(config)


def discriminator_config_inception():
    config1x = Config({
        "n_enc_seq": 56 * 56,
        "n_dec_seq": 56 * 56,
        "n_layer": 1,
        "d_hidn": 128,
        "i_pad": 0,
        "d_ff": 1024,
        "d_MLP_head": 128,
        "n_head":4,
        "d_head": 128,
        "dropout": 0.1,
        "emb_dropout": 0.1,
        "layer_norm_epsilon": 1e-2,
        "n_output": 1
    })
    config2x = Config({
        "n_enc_seq": 25 * 25,
        "n_dec_seq": 25 * 25,
        "n_layer": 1,
        "d_hidn": 128,
        "i_pad": 0,
        "d_ff": 1024,
        "d_MLP_head": 128,
        "n_head":4,
        "d_head": 128,
        "dropout": 0.1,
        "emb_dropout": 0.1,
        "layer_norm_epsilon": 1e-2,
        "n_output": 1
    })
    config3x = Config({
        "n_enc_seq": 25 * 25,
        "n_dec_seq": 25 * 25,
        "n_layer": 1,
        "d_hidn": 128,
        "i_pad": 0,
        "d_ff": 1024,
        "d_MLP_head": 128,
        "n_head":4,
        "d_head": 128,
        "dropout": 0.1,
        "emb_dropout": 0.1,
        "layer_norm_epsilon": 1e-2,
        "n_output": 1
    })
    config4x = Config({
        "n_enc_seq": 7 * 7,
        "n_dec_seq": 7 * 7,
        "n_layer": 1,
        "d_hidn": 128,
        "i_pad": 0,
        "d_ff": 1024,
        "d_MLP_head": 128,
        "n_head":4,
        "d_head": 128,
        "dropout": 0.1,
        "emb_dropout": 0.1,
        "layer_norm_epsilon": 1e-2,
        "n_output": 1
    })

    return config1x, config2x, config3x, config4x
