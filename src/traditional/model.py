# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os
from gensim import models, corpora
from annoy import AnnoyIndex

from src import utils


class SentSimModel(object):
    def __init__(self, model_dir='', num_topics=None, num_keywords=None, num_trees=10):
        self._allowed_modes = ['tfidf', 'lda']
        self.dictionary = {mode: None for mode in self._allowed_modes}
        self.corpus = {mode: None for mode in self._allowed_modes}
        self.models = {mode: None for mode in self._allowed_modes}
        self.annoys = {mode: None for mode in self._allowed_modes}
        self.model_dir = model_dir
        self.num_topics = num_topics
        self.num_keywords = num_keywords
        self.num_trees = num_trees

    def _fit(self, list_of_tokens, mode):
        utils.verbose('start training {} dictionary'.format(mode))
        self.dictionary[mode] = corpora.Dictionary(list_of_tokens)
        utils.verbose('start building {} corpus'.format(mode))
        self.corpus[mode] = [
            self.dictionary[mode].doc2bow(toks) for toks in list_of_tokens]
        utils.verbose('start training {} model'.format(mode))
        if mode == 'lda':
            self.models['lda'] = models.LdaMulticore(
                self.corpus['lda'], self.num_topics, id2word=self.dictionary['lda'])
        elif mode == 'tfidf':
            self.models['tfidf'] = models.TfidfModel(self.corpus['tfidf'])
        else:
            raise ValueError('Invalid mode: {}'.format(mode))
        utils.verbose('start saving {} dictionary and model'.format(mode))
        self.models[mode].save(os.path.join(self.model_dir, '{}.pkl'.format(mode)))
        self.dictionary[mode].save(os.path.join(self.model_dir, '{}.dict'.format(mode)))
        self._fit_vec(list_of_tokens, mode)

    def _fit_vec(self, list_of_tokens, mode):
        ann_path = os.path.join(self.model_dir, '{}.ann'.format(mode))
        if mode == 'tfidf':
            self.annoys[mode] = AnnoyIndex(self.num_keywords)
        elif mode == 'lda':
            self.annoys[mode] = AnnoyIndex(self.num_topics)
        else:
            raise ValueError('Invalid mode: {}'.format(mode))
        for n, toks in enumerate(list_of_tokens):
            if not n % 10000 and n:
                utils.verbose(
                    'processing vectorization for {} lines in {} mode'.format(n, mode))
            if mode == 'tfidf':
                vec = self.get_tfidf(toks)
            elif mode == 'lda':
                vec = self.get_lda(toks)
            else:
                raise ValueError('Invalid mode: {}'.format(mode))
            self.annoys[mode].add_item(n, vec)
        self.annoys[mode].build(10)
        self.annoys[mode].save(ann_path)
        utils.verbose('dump {} annoy into {}'.format(mode, ann_path))

    def _get(self, toks, mode):
        vec_bow = self.dictionary[mode].doc2bow(toks)
        if mode == 'lda':
            vec = [i[1] for i in self.models['lda'].get_document_topics(vec_bow)]
        elif mode == 'tfidf':
            vec = [0] * self.num_keywords
            resu = self.models['tfidf'][vec_bow]
            for x, y in resu:
                vec[x] = y
        else:
            raise ValueError('invalid mode: {}'.format(mode))
        return vec

    def _load(self, mode):
        path_model = os.path.join(self.model_dir, '{}.pkl'.format(mode))
        path_dict = os.path.join(self.model_dir, '{}.dict'.format(mode))
        path_ann = os.path.join(self.model_dir, '{}.ann'.format(mode))
        if all([os.path.exists(i) for i in [path_model, path_dict, path_ann]]):
            if mode == 'lda':
                self.models[mode] = models.LdaMulticore.load(path_model)
            elif mode == 'tfidf':
                self.models[mode] = models.TfidfModel.load(path_model)
            else:
                raise ValueError('Invalid mode: {}'.format(mode))
            utils.verbose('load {} model from {}'.format(mode, path_model))
            self.dictionary[mode] = corpora.Dictionary.load(path_dict)
            utils.verbose('load {} dictionary from {}'.format(mode, path_dict))
            if mode == 'lda':
                self.annoys[mode] = AnnoyIndex(self.num_topics)
            elif mode == 'tfidf':
                self.annoys[mode] = AnnoyIndex(self.num_keywords)
            else:
                raise ValueError('Invalid mode: {}'.format(mode))
            self.annoys[mode].load(path_ann)
            utils.verbose('load {} annoy from {}'.format(mode, path_ann))
        else:
            raise ValueError('File under directory {} is not found'.format(self.model_dir))

    def fit_lda(self, list_of_tokens):
        self._fit(list_of_tokens, 'lda')

    def fit_tfidf(self, list_of_tokens):
        self._fit(list_of_tokens, 'tfidf')

    def get_lda(self, toks):
        return self._get(toks, 'lda')

    def get_tfidf(self, toks):
        return self._get(toks, 'tfidf')

    def load_lda(self):
        self._load('lda')

    def load_tfidf(self):
        self._load('tfidf')

    def search_tfidf(self, toks, num):
        return self.annoys['tfidf'].get_nns_by_vector(self.get_tfidf(toks), num)

    def search_lda(self, toks, num):
        return self.annoys['lda'].get_nns_by_vector(self.get_lda(toks), num)
