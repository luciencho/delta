# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os
import argparse

from src import utils
from src.retrieval import model
from src.retrieval import train


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tmp_dir', type=str,
                        required=True, help='tmp_dir')
    parser.add_argument('--model_dir', type=str,
                        required=True, help='model_dir')
    parser.add_argument('--hparams', type=str,
                        required=True, help='hparam_set')
    parser.add_argument('--gpu_device', type=str,
                        default='0', help='gpu_device')
    parser.add_argument('--gpu_memory', type=float,
                        default=0.23, help='gpu_memory_fraction')
    args = parser.parse_args()
    if args.hparams == 'solo_base':
        original = model.solo_base()
    else:
        raise ValueError('Unknown hparams: {}'.format(args.hparams))
    for k, v in original.__dict__.items():
        if not k.startswith('_'):
            utils.verbose('add attribute {} [{}] to hparams'.format(k, v))
            setattr(args, k, v)
    args.path = {'model': os.path.join(args.model_dir, 'retrieval', 'model'),
                 'vocab': [os.path.join(args.tmp_dir, '{}.vcb'.format(i)) for i in [
                     args.word_size, args.char_size]],
                 'train_x': os.path.join(args.tmp_dir, 'train_q.txt'),
                 'train_y': os.path.join(args.tmp_dir, 'train_a.txt'),
                 'dev_x': os.path.join(args.tmp_dir, 'dev_q.txt'),
                 'dev_y': os.path.join(args.tmp_dir, 'dev_a.txt')}
    return args


if __name__ == '__main__':
    hparams = get_args()
    utils.verbose('Start training')
    train.process(hparams)
    utils.verbose('Finish training')
