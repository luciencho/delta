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


def train_keywords(data, path):
    vocab_counter = {}
    i = 0
    for line in data:
        for word in line:
            if word in vocab_counter:
                vocab_counter[word] += 1
            else:
                vocab_counter[word] = 1
        if not i % 10000 and i:
            utils.verbose('processing {} lines'.format(i))
        i += 1
    with open(path, 'w', encoding='utf-8') as f:
        for key, value in vocab_counter.items():
            f.write(str(key) + ' ' + str(math.log(i / value, 2)) + '\n')
    utils.verbose('keywords are saved in {}'.format(path))
