# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os
import time
import shutil
import argparse

from src.retrieval import model


def verbose(line):
    print('[{}]\t{}'.format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), line))


def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def clean_and_make_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
        time.sleep(5)
    os.makedirs(directory)


def raise_inexistence(path):
    if not os.path.exists(path):
        raise ValueError('directory or path {} does not exist'.format(
            os.path.abspath(path)))


def read_lines(path):
    raise_inexistence(path)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip().split('\n')


def write_lines(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def write_result(args, loss, model_dir):
    lines = []
    file_path = os.path.join(model_dir, '{}.{}'.format(int(time.time()), 'rst'))
    for k, v in args.__dict__.items():
        if not k.startswith('_'):
            lines.append('hparams.{}: [{}]'.format(k, v))
            verbose('hparams.{}: [{}]'.format(k, v))
    lines.append('lowest loss: [{}]'.format(loss))
    verbose('lowest loss: [{}]'.format(loss))
    write_lines(file_path, lines)


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
    if args.hparams == 'solo_lstm':
        original = model.solo_lstm()
    elif args.hparams == 'solo_lstm_v1':
        original = model.solo_lstm_v1()
    elif args.hparams == 'solo_gru':
        original = model.solo_gru()
    else:
        raise ValueError('Unknown hparams: {}'.format(args.hparams))
    for k, v in original.__dict__.items():
        if not k.startswith('_'):
            verbose('add attribute {} [{}] to hparams'.format(k, v))
            setattr(args, k, v)
    args.path = {'model': os.path.join(args.model_dir, 'retrieval', 'model'),
                 'vocab': [os.path.join(args.tmp_dir, '{}.vcb'.format(i)) for i in [
                     args.word_size, args.char_size]],
                 'train_x': os.path.join(args.tmp_dir, 'train_q.txt'),
                 'train_y': os.path.join(args.tmp_dir, 'train_a.txt'),
                 'dev_x': os.path.join(args.tmp_dir, 'dev_q.txt'),
                 'dev_y': os.path.join(args.tmp_dir, 'dev_a.txt'),
                 'ann': os.path.join(args.model_dir, 'retrieval', 'ann')}
    return args


def data_gen_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='data_dir')
    parser.add_argument('--tmp_dir', type=str, help='tmp_dir')
    parser.add_argument('--hparams', type=str, help='hparam_set')
    args = parser.parse_args()
    if args.hparams == 'solo_lstm':
        original = model.solo_lstm()
    elif args.hparams == 'solo_lstm_v1':
        original = model.solo_lstm_v1()
    elif args.hparams == 'solo_gru':
        original = model.solo_gru()
    else:
        raise ValueError('Unknown hparams: {}'.format(args.hparams))
    for k, v in original.__dict__.items():
        if not k.startswith('_'):
            verbose('add attribute {} [{}] to hparams'.format(k, v))
            setattr(args, k, v)
    return args
