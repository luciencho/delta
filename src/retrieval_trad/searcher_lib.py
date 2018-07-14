# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from src import utils
from src.retrieval_trad.keyword_lib import load_keywords
from src.retrieval_trad.model import SentSimModel
from src.data_utils.vocab import Tokenizer


class Searcher(object):
    def __init__(self, args):
        tokenizer = Tokenizer(args.path['vocab'])
        keywords = load_keywords(args.path['keyword'])
        self.ss_model = SentSimModel(tokenizer, keywords, args.tfidf_vocab_size)
        self.ss_model.set_data(utils.read_lines(args.path['train_x']))
        self.ss_model.simple_data()
        self.ss_model.load(args.path['tfidf'], 'tfidf')
        self.ss_model.apply_model('tfidf')

    def search_line(self, line, mode='tfidf', num=15):
        candidate_ids = self.ss_model.ranker(line, mode, num)
        return candidate_ids
