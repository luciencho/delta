# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from src import utils
from src.dual_encoder import trainer_lib as retrieval_trainer_lib
from src.tfidf import trainer_lib as retrieval_trad_trainer_lib
from src.tradictional import keyword

trainer_index = {
    'dual_encoder': retrieval_trainer_lib,
    'tfidf': retrieval_trad_trainer_lib,
    'keyword': keyword}


if __name__ == '__main__':
    hparams = utils.get_args()
    if hparams.problem is None:
        raise ValueError('At least one problem must be announced')
    elif hparams.problem not in trainer_index:
        raise ValueError('Invalid problem: {}'.format(hparams.problem))
    else:
        utils.verbose('Start training problem: {}'.format(hparams.problem))
        trainer_index[hparams.problem].process(hparams)
        utils.verbose('Finish training problem: {}'.format(hparams.problem))
