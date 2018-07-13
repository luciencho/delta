# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import sys
import os

from src import utils
from src.retrieval import searcher_lib as retrieval_searcher_lib
from src.retrieval_trad import searcher_lib as retrieval_trad_searcher_lib


args = utils.get_args(use_fake=True)
retrieval_searcher = retrieval_searcher_lib.Searcher(args)
retrieval_trad_searcher = retrieval_trad_searcher_lib.Searcher(args)


def run_prediction(in_path, out_path):
    with open(in_path, 'r', encoding='utf-8') as f:
        resu = f.read().strip().split('\n')
    ret = []
    for i in resu:
        ret.append(retrieval_searcher.search_line(i, 1)[0])
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(ret))


if __name__ == '__main__':
    input_file_path = None
    output_file_path = None
    try:
        input_file_path = sys.argv[1]
        output_file_path = sys.argv[2]
    except IndexError:
        print('index error')
        exit(0)
    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    run_prediction(input_file_path, output_file_path)
