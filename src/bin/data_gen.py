# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from src import utils
from src.data_utils import data_generator
from src.data_utils import vocab_generator


if __name__ == '__main__':
    hparams = utils.data_gen_args()
    utils.verbose('Start generating data')
    data_generator.process(hparams)
    utils.verbose('Finish generating data')
    utils.verbose('Start generating vocab')
    vocab_generator.process(hparams)
    utils.verbose('Finish generating vocab')
