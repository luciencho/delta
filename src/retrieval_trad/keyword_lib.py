# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import math

from src import utils
from src.data_utils.vocab import Tokenizer


def load_keywords(path_keyword):
    idf_freq = {}
    utils.verbose('loading keywords from {}'.format(path_keyword))
    with open(path_keyword, 'r', encoding='utf-8') as f:
        for line in f:
            word, freq = line.strip().split(' ')
            idf_freq[int(word)] = float(freq)
    keywords = sorted(idf_freq, key=idf_freq.get)
    return keywords


def process(args):
    utils.make_directory(args.path['model'])
    tokenizer = Tokenizer(args.path['vocab'])
    train_x_enc = [tokenizer.encode_line_trad(i) for i in utils.read_lines(
        args.path['train_x'])]
    train_y_enc = [tokenizer.encode_line_trad(i) for i in utils.read_lines(
        args.path['train_y'])]
    trainset = train_x_enc + train_y_enc
    vocab_counter = {}

    i = 0
    for line in trainset:
        for word in line:
            if word in vocab_counter:
                vocab_counter[word] += 1
            else:
                vocab_counter[word] = 1
        if not i % 10000 and i:
            utils.verbose('processing {} lines'.format(i))
        i += 1
    with open(args.path['keyword'], 'w', encoding='utf-8') as f:
        for key, value in vocab_counter.items():
            f.write(str(key) + ' ' + str(math.log(i / value, 2)) + '\n')
    utils.verbose('keywords are saved in {}'.format(args.path['keyword']))
