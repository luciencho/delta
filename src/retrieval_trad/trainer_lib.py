# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from src import utils
from src.retrieval_trad.keyword_lib import load_keywords
from src.retrieval_trad.model import SentSimModel
from src.data_utils.vocab import Tokenizer


def process(args):
    utils.make_directory(args.path['model'])
    tokenizer = Tokenizer(args.path['vocab'])
    keywords = load_keywords(args.path['keyword'])
    ss_model = SentSimModel(tokenizer, keywords, args.tfidf_vocab_size)
    ss_model.set_data(utils.read_lines(args.path['train_x']))
    ss_model.simple_data()
    ss_model.fit(args.path['tfidf'], 'tfidf')
