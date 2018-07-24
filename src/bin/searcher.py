# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

# import sys
# import os

from src import utils
from src.dual_encoder.searcher_lib import Searcher as DualEncoderSearcher
from src.tradictional.searcher_lib import Searcher as TraditionalSearcher


class Inspector(object):
    def __init__(self, args):
        self.dual_encoder_searcher = DualEncoderSearcher(args)
        self.traditional_searcher = TraditionalSearcher(args)
        self.questions = utils.read_lines(args.path['train_x'])
        self.answers = utils.read_lines(args.path['train_y'])

    def _view_candidates(self, search_fn, line, num_keeps=50):
        ids = search_fn(line, num_keeps)
        print('{}\n{}\nlda:'.format('-+' * 35, line))
        for idx in ids:
            print('id: {}'.format(idx))
            print('question: {}'.format(self.questions[idx]))
            print('answer: {}'.format(self.answers[idx]))

    def view_lda_candidates(self, line, num_keeps=50):
        self._view_candidates(self.traditional_searcher.search_by_lda, line, num_keeps)

    def view_de_candidates(self, line, num_keeps=50):
        self._view_candidates(self.dual_encoder_searcher.search_line, line, num_keeps)

    def view_tfidf_candidates(self, line, num_keeps=50):
        self._view_candidates(self.traditional_searcher.search_by_tfidf, line, num_keeps)


# def run_prediction(in_path, out_path):
#     with open(in_path, 'r', encoding='utf-8') as f:
#         lines = f.read().strip().split('\n')
#     ret = []
#     for line in lines:
#         lda_candidate_ids = traditional_searcher.search_by_lda(line, 50)
#         dual_encoder_candidate_ids = dual_encoder_searcher.search_line(line, 50)
#         tfidf_candidate_ids = traditional_searcher.search_by_tfidf(line, 50)
#     with open(out_path, 'w', encoding='utf-8') as f:
#         f.write('\n'.join(ret))
#
#
# if __name__ == '__main__':
#     input_file_path = None
#     output_file_path = None
#     try:
#         input_file_path = sys.argv[1]
#         output_file_path = sys.argv[2]
#     except IndexError:
#         print('index error')
#         exit(0)
#     output_dir = os.path.dirname(output_file_path)
#     if not os.path.exists(output_dir) and output_dir:
#         os.makedirs(output_dir)
#     inspector = Inspector(utils.get_args(use_fake=True))
#     run_prediction(input_file_path, output_file_path)
