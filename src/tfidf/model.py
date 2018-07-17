# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os
import shutil
import time
from gensim import corpora, models, similarities

from src import utils


class SentSimModel(object):
    def __init__(self, tokenizer, keywords, vocab_size=2000):
        self.tokenizer = tokenizer
        self.keywords = keywords
        self.vocab_size = vocab_size
        self.lines_1 = None
        self.dictionary = None
        self.corpus_simple = None
        self.corpus = None
        self.models = {}
        self.index = None

    def set_data(self, lines_1):
        self.lines_1 = lines_1

    def simple_data(self):
        if self.lines_1 is None:
            raise ValueError('set data is a must before this step')
        texts = [[str(s) for s in self.tokenizer.encode_line_into_words(line)
                  if s in self.keywords[: self.vocab_size]] for line in self.lines_1]
        self.dictionary = corpora.Dictionary(texts)
        self.corpus_simple = [self.dictionary.doc2bow(text) for text in texts]

    def fit(self, model_path, mode='tfidf'):
        if self.dictionary is None:
            raise ValueError('simple data is a must before this step')
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            time.sleep(5)
        self.models[mode] = models.TfidfModel(self.corpus_simple)
        self.models[mode].save(model_path)

    def load(self, model_path, mode='tfidf'):
        utils.verbose('loading model from model_dir')
        self.models[mode] = models.TfidfModel.load(model_path)

    def apply_model(self, mode='tfidf'):
        self.corpus = self.models[mode][self.corpus_simple]
        self.index = similarities.MatrixSimilarity(self.corpus)

    def line2vec(self, line, mode='tfidf'):
        words = [str(s) for s in self.tokenizer.encode_line_into_words(line)
                 if s in self.keywords[: self.vocab_size]]
        vec_bow = self.dictionary.doc2bow(words)
        return self.models[mode][vec_bow]

    def ranker(self, line, mode='tfidf', num=50):
        sentence_vec = self.line2vec(line, mode)
        sims = self.index[sentence_vec]
        sim_sort = sorted(list(enumerate(sims)), key=lambda item: item[1], reverse=True)
        tops = sim_sort[0: num]
        return [i[0] for i in tops]
