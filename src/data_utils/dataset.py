# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import random


class SoloBatch(object):
    def __init__(self, tokenizer, max_lens):
        self.tokenizer = tokenizer
        self.max_lens = max_lens
        self.epoch = 1
        self.pairs = []

    def set_data(self, lines_1, lines_2):
        assert len(lines_1) == len(lines_2)
        self.pairs = list(zip(lines_1, lines_2))

    @property
    def data_size(self):
        return len(self.pairs)

    def shuffle_data(self):
        random.shuffle(self.pairs)

    def _encode_line(self, line, max_len):
        wc_pairs = self.tokenizer.encode_line(line)[: max_len]
        wc_pairs += [(self.tokenizer.PAD_ID,
                      self.tokenizer.PAD_ID)] * (max_len - len(wc_pairs))
        return wc_pairs

    def _next_ids(self, interval, idx):
        start = idx % self.data_size
        end = start % self.data_size + interval
        if end > self.data_size:
            ids = list(range(start, self.data_size)) + \
                  list(range(0, end - self.data_size))
            flag = True
        else:
            ids = list(range(start, end))
            flag = False
        return ids, flag

    def next_batch(self, batch_size, idx):
        ids, update_epoch = self._next_ids(batch_size, idx)
        idx += batch_size
        pairs = [self.pairs[i] for i in ids]
        lines_1, lines_2 = zip(*pairs)
        input_x = [self._encode_line(i, self.max_lens[0]) for i in lines_1]
        input_y = [self._encode_line(i, self.max_lens[1]) for i in lines_2]
        if update_epoch:
            self.epoch += 1
            self.shuffle_data()
        return input_x, input_y, idx, update_epoch
