# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os
import argparse
import platform

from src import utils
from src.dual_encoder import model
from src.data_utils import data
from src.data_utils.vocab import Tokenizer
from src.utils import hparams_utils


def _major_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tmp_dir', type=str, required=True, help='tmp_dir')
    parser.add_argument('--model_dir', type=str, required=True, help='model_dir')
    parser.add_argument('--hparams', type=str, required=True, help='hparam_set')
    parser.add_argument('--gpu_device', type=str, default='0', help='gpu_device')
    parser.add_argument('--gpu_memory', type=float, default=0.23, help='gpu_memory_fraction')
    parser.add_argument('--problem', type=str, required=False, help='problem')
    args = parser.parse_args()
    return args


def _minor_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='data_dir')
    parser.add_argument('--tmp_dir', type=str, help='tmp_dir')
    parser.add_argument('--hparams', type=str, help='hparam_set')
    args = parser.parse_args()
    return args


def _fake_args():

    class FakeArgs(object):
        if platform.system() == 'Windows':
            tmp_dir = r'E:\competition\jddc_v2\tmp_belta'
            model_dir = r'E:\competition\jddc_v2\models'
        else:
            tmp_dir = r'/submission/tmp'
            model_dir = r'/submission/models'
        hparams = 'penta_lstm'

    return FakeArgs()


def _reconstruct_args(args):
    hp_mode, hparams = args.hparams.split('_')
    args.tokenizer = Tokenizer

    if hparams == 'lstm':
        original = hparams_utils.lstm()
    elif hparams == 'gru':
        original = hparams_utils.gru()
    elif hparams == 'lstmln':
        original = hparams_utils.lstm_ln()
    elif hparams == 'lstmrcnn':
        original = hparams_utils.lstm_rcnn()
    else:
        raise ValueError('Unknown hparams: {}'.format(hparams))

    if hp_mode == 'solo':
        args.batch = data.SoloBatch
        args.model = model.SoloModel
        args.max_lens = [original.x_max_len, original.y_max_len]
    elif hp_mode == 'penta':
        args.batch = data.PentaBatch
        args.model = model.PentaModel
        args.max_lens = [original.y_max_len, original.y_max_len]
    else:
        raise ValueError('Unknown hp_mode: {}'.format(hp_mode))

    for k, v in original.__dict__.items():
        if not k.startswith('_'):
            utils.verbose('add attribute {} [{}] to hparams'.format(k, v))
            setattr(args, k, v)
    return args


def major_args(use_fake=False):
    """ args for trainer & searcher

    :param use_fake: bool
    :return: args
    """
    args = _reconstruct_args(_fake_args() if use_fake else _major_args())
    args.path = {'model': os.path.join(args.model_dir, args.hparams, 'model'),
                 'vocab': [os.path.join(args.tmp_dir, '{}.vcb'.format(i)) for i in [
                     args.word_size, args.char_size]],
                 'train_x': os.path.join(args.tmp_dir, 'train_q.txt'),
                 'train_y': os.path.join(args.tmp_dir, 'train_a.txt'),
                 'dev_x': os.path.join(args.tmp_dir, 'dev_q.txt'),
                 'dev_y': os.path.join(args.tmp_dir, 'dev_a.txt'),
                 'ann': os.path.join(args.model_dir, args.hparams, 'dual_encoder.ann')}
    return args


def minor_args(use_fake=False):
    """ args for data generation & vocab generation

    :param use_fake: bool
    :return: args
    """
    args = _reconstruct_args(_fake_args() if use_fake else _minor_args())
    return args
