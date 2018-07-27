# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import sys
import os

from src import utils
from src.utils import args_utils
from src.utils import searcher_lib


class Inspector(object):
    def __init__(self, args):
        self.dual_encoder_searcher = searcher_lib.DualEncoderSearcher(args)
        self.lda_searcher = searcher_lib.LDASearcher(args)
        self.tfidf_searcher = searcher_lib.TFIDFSearcher(args)
        self.questions = utils.read_lines(args.path['train_x'])
        self.answers = utils.read_lines(args.path['train_y'])

    def _view_candidates(self, resu, line, mode):
        print('{}\n{}'.format('-+' * 35, line))
        for idx, sim in resu:
            print('id: {} | mode: {} | sim: {}'.format(idx, mode, sim))
            print('question: {}'.format(self.questions[idx]))
            print('answer: {}'.format(self.answers[idx]))

    def view_lda_candidates(self, line, num_keeps=50):
        resu = self.lda_searcher.search_line(line, num_keeps)
        self._view_candidates(resu, line, 'lda')

    def view_de_candidates(self, line, num_keeps=50):
        resu = self.dual_encoder_searcher.search_line(line, num_keeps)
        self._view_candidates(resu, line, 'dual_encoder')

    def view_tfidf_candidates(self, line, num_keeps=50):
        resu = self.tfidf_searcher.search_line(line, num_keeps)
        self._view_candidates(resu, line, 'tfidf')

    def _vote(self, line, weights=None, num_keeps=50, num_consider=None):
        if weights is None:
            weights = [0.45, 0.45, 0.1]
        else:
            assert sum(weights) == 1
        if num_consider is None:
            num_consider = max(num_keeps, 50)
        res_de = self.dual_encoder_searcher.search_line(line, num_consider)
        res_tfidf = self.tfidf_searcher.search_line(line, num_consider)
        res_lda = self.lda_searcher.search_line(line, num_consider)
        collect_de = {k: v for k, v in res_de}
        collect_tfidf = {k: v for k, v in res_tfidf}
        collect_lda = {k: v for k, v in res_lda}

        ids = set([i[0] for i in res_de + res_tfidf + res_lda])
        resu = [(i, sum([weights[0] * collect_de.get(i, 0), weights[1] * collect_tfidf.get(i, 0),
                         weights[2] * collect_lda.get(i, 0)])) for i in ids]
        resu = sorted(resu, key=lambda x: x[1], reverse=True)
        return resu

    def view_all(self, line, weights=None, num_keeps=50, num_consider=None):
        resu = self._vote(line, weights, num_keeps, num_consider)
        self._view_candidates(resu[: num_keeps], line, 'mix: {}'.format(weights))

    def one(self, line, weights=None, num_consider=50):
        resu = self._vote(line, weights, num_consider, num_consider)
        return self.answers[resu[0][0]]


def run_prediction(in_path, out_path):
    with open(in_path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    ret = []
    for line in lines:
        ret.append(ins.one(line))
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(ret))


if __name__ == '__main__':
    ins = Inspector(args_utils.major_args(True))
    input_file_path = None
    output_file_path = None
    try:
        input_file_path = sys.argv[1]
        output_file_path = sys.argv[2]
    except IndexError:
        print('index error')
        exit(0)
    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir) and output_dir:
        os.makedirs(output_dir)
    run_prediction(input_file_path, output_file_path)
