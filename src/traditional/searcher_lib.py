# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from src.data_utils.vocab import Tokenizer
from src.traditional.keyword import load_keywords
from src.traditional.model import SentSimModel


class Searcher(object):
    def __init__(self, args):
        self.tokenizer = Tokenizer(args.path['vocab'])
        self.keywords = load_keywords(args.path['model'])
        self.num_keywords = args.num_keywords
        self.num_trees = args.num_trees
        self.num_topics = args.num_topics
        self.model = SentSimModel(
            args.path['model'], self.num_topics, self.num_keywords, self.num_trees)
        self.model.load_tfidf()
        self.model.load_lda()

    def search_by_lda(self, line, num=15):
        toks = [str(s) for s in self.tokenizer.encode_line_into_words(line)]
        return self.model.search_lda(toks, num)

    def search_by_tfidf(self, line, num=15):
        toks = [str(s) for s in self.tokenizer.encode_line_into_words(line)
                if s in self.keywords[: self.num_keywords]]
        return self.model.search_tfidf(toks, num)
