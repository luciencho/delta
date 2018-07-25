# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def lstm():
    hparams = tf.contrib.training.HParams(
        num_keywords=10000,
        num_trees=10,
        word_size=50000,
        char_size=5000,
        emb_dim=256,
        hidden=128,
        num_layers=1,
        rnn_type='lstm',
        use_layer_norm=False,
        l2_weight=1e-4,
        learning_rate=5e-3,
        decay_rate=0.98,
        keep_prob=0.6,
        max_steps=20000,
        show_steps=100,
        save_steps=500,
        batch_size=256,
        x_max_len=80,
        y_max_len=40,
        num_topics=128)
    return hparams


def gru():
    hparams = lstm()
    hparams.rnn_type = 'gru'
    return hparams


def lstm_ln():
    hparams = lstm()
    hparams.use_layer_norm = True
    return hparams
