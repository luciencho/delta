# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import argparse

from src import utils
from src.retrieval import model
from src.data_utils import data_generator
from src.data_utils import vocab_generator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='data_dir')
    parser.add_argument('--tmp_dir', type=str, help='tmp_dir')
    parser.add_argument('--hparams', type=str, help='hparam_set')
    args = parser.parse_args()
    if args.hparams == 'solo_base':
        original = model.solo_base()
    else:
        raise ValueError('Unknown hparams: {}'.format(args.hparams))
    for k, v in original.__dict__.items():
        if not k.startswith('_'):
            utils.verbose('add attribute {} [{}] to hparams'.format(k, v))
            setattr(args, k, v)
    return args


if __name__ == '__main__':
    hparams = get_args()
    utils.verbose('Start generating data')
    data_generator.process(hparams)
    utils.verbose('Finish generating data')
    utils.verbose('Start generating vocab')
    vocab_generator.process(hparams)
    utils.verbose('Finish generating vocab')
