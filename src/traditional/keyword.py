# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os
import math

from src import utils


def load_keywords(model_dir):
    path = os.path.join(model_dir, 'keywords.txt')
    idf_freq = {}
    utils.verbose('loading keywords from {}'.format(path))
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            word, freq = line.strip().split(' ')
            idf_freq[int(word)] = float(freq)
    keywords = sorted(idf_freq, key=idf_freq.get)
    return keywords


def train_keywords(data, model_dir):
    path = os.path.join(model_dir, 'keywords.txt')
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
