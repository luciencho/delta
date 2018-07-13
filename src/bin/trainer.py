# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from src import utils
from src.retrieval import trainer_lib as retrieval_trainer_lib
from src.retrieval_trad import trainer_lib as retrieval_trad_trainer_lib
from src.retrieval_trad import keyword_lib


trainer_index = {
    'retrieval': retrieval_trainer_lib,
    'retrieval_trad': retrieval_trad_trainer_lib,
    'keyword': keyword_lib}


if __name__ == '__main__':
    hparams = utils.get_args()
    if hparams.problems is None:
        raise ValueError('At least one problem must be announced')
    for p in hparams.problems:
        if p not in trainer_index:
            raise ValueError('Invalid problem: {}'.format(p))
        utils.verbose('Start training problem: {}'.format(p))
        trainer_index[p].process(hparams)
        utils.verbose('Finish training problem: {}'.format(p))
