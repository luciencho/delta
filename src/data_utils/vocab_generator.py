# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from os.path import join
from src import utils
from src.data_utils import vocab


def process(hparam):
    utils.raise_inexistence(hparam.tmp_dir)
    tokenizer = vocab.Tokenizer()
    all_data = []
    paths = [join(hparam.tmp_dir, i) for i in [
        'train_q.txt', 'train_a.txt', 'dev_q.txt', 'dev_a.txt']]
    word_path = join(hparam.tmp_dir, '{}.vcb'.format(hparam.word_size))
    char_path = join(hparam.tmp_dir, '{}.vcb'.format(hparam.char_size))
    for path in paths:
        utils.raise_inexistence(path)
        all_data += utils.read_lines(path)
    tokenizer.build_vocab(
        all_data, [hparam.word_size, hparam.char_size], [word_path, char_path])
