# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from src import utils
from src.retrieval_trad.model import SentSimModel
from src.data_utils.vocab import Tokenizer


def process(args):
    tokenizer = Tokenizer(args.path['vocab'])
    ss_model = SentSimModel(tokenizer, args.tfidf_vocab_size)
    ss_model.set_data(utils.read_lines(args.path['train_x']))
    ss_model.simple_data()
    ss_model.fit(args.path['tfidf'], 'tfidf')
