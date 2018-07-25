# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import random


class Batch(object):
    def __init__(self, tokenizer, max_lens):
        self.tokenizer = tokenizer
        self.max_lens = max_lens
        self.epoch = 1
        self.pairs = []

    def set_data(self, lines_x, lines_y):
        assert len(lines_x) == len(lines_y)
        self.pairs = list(zip(lines_x, lines_y))

    @property
    def data_size(self):
        return len(self.pairs)

    def shuffle_data(self):
        random.shuffle(self.pairs)

    def _encode_line(self, line, max_len):
        wc_pairs = self.tokenizer.encode_line_into_pairs(line)[: max_len]
        wc_pairs += [(self.tokenizer.PAD_ID,
                      self.tokenizer.PAD_ID)] * (max_len - len(wc_pairs))
        return wc_pairs

    def encode_x(self, line):
        raise NotImplementedError()

    def encode_y(self, line):
        raise NotImplementedError()

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
        lines_x, lines_y = zip(*pairs)
        input_x = [self.encode_x(i) for i in lines_x]
        input_y = [self.encode_y(i) for i in lines_y]
        if update_epoch:
            self.epoch += 1
            self.shuffle_data()
        return input_x, input_y, idx, update_epoch


class SoloBatch(Batch):
    def __init__(self, tokenizer, max_lens):
        super(SoloBatch, self).__init__(tokenizer, max_lens)

    def encode_x(self, line):
        return self._encode_line(line, self.max_lens[0])

    def encode_y(self, line):
        return self._encode_line(line, self.max_lens[1])


class PentaBatch(Batch):
    def __init__(self, tokenizer, max_lens):
        super(PentaBatch, self).__init__(tokenizer, max_lens)

    def encode_x(self, line):
        return [self._encode_line(l, self.max_lens[0]) for l in line.split('<s>') if l]

    def encode_y(self, line):
        return self._encode_line(line, self.max_lens[1])
