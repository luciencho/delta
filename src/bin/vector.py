# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from src import utils
from src.retrieval import vector_lib


if __name__ == '__main__':
    hparams = utils.get_args()
    utils.verbose('Start training')
    vector_lib.process(hparams)
    utils.verbose('Finish training')
