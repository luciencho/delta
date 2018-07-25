# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os
from gensim import models, corpora, similarities
from annoy import AnnoyIndex

from src import utils


class SentSimModel(object):
    def __init__(self, args):
        self.model_dir = args.path['model']
        self.dict = None
        self.corpus = None
        self.model = None

    def fit(self, list_toks):
        raise NotImplementedError()

    def get(self, toks):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def search(self, toks, num):
        raise NotImplementedError()


class LDAModel(SentSimModel):
    def __init__(self, args):
        super(LDAModel, self).__init__(args)
        self.vec_dim = args.num_topics
        self.num_trees = args.num_trees
        self.ann = None
        self.paths = dict(model=os.path.join(self.model_dir, 'lda.pkl'),
                          dict=os.path.join(self.model_dir, 'lda.dict'),
                          ann=os.path.join(self.model_dir, 'lda.ann'))

    def fit(self, list_toks):
        utils.verbose('start training lda dictionary')
        self.dict = corpora.Dictionary(list_toks)

        utils.verbose('start building lda corpus')
        self.corpus = [self.dict.doc2bow(toks) for toks in list_toks]

        utils.verbose('start training lda model')
        self.model = models.LdaMulticore(self.corpus, self.vec_dim, id2word=self.dict)

        utils.verbose('start saving lda dictionary and model')
        self.model.save(self.paths['model'])
        self.dict.save(self.paths['dict'])

        utils.verbose('start vectorization for lda')
        self.ann = AnnoyIndex(self.vec_dim)
        for n, toks in enumerate(list_toks):
            if not n % 10000 and n:
                utils.verbose('vectorizing {} lines for lda'.format(n))
            vec = self.get(toks)
            self.ann.add_item(n, vec)

        utils.verbose('start building lda ann')
        self.ann.build(self.num_trees)
        self.ann.save(self.paths['ann'])
        utils.verbose('dump lda annoy into {}'.format(self.paths['ann']))

    def get(self, toks):
        vec_bow = self.dict.doc2bow(toks)
        vec = [0] * self.vec_dim
        resu = self.model.get_document_topics(vec_bow)
        for x, y in resu:
            vec[x] = y
        return vec

    def load(self):
        if all([os.path.exists(i) for i in self.paths.values()]):
            self.model = models.LdaMulticore.load(self.paths['model'])
            utils.verbose('load lda model from {}'.format(self.paths['model']))
            self.dict = corpora.Dictionary.load(self.paths['dict'])
            utils.verbose('load lda dictionary from {}'.format(self.paths['dict']))
            self.ann = AnnoyIndex(self.vec_dim)
            self.ann.load(self.paths['ann'])
            utils.verbose('load lda annoy from {}'.format(self.paths['ann']))
        else:
            raise ValueError('Files under directory {} disappear'.format(self.model_dir))

    def search(self, toks, num):
        return self.ann.get_nns_by_vector(self.get(toks), num)


class TFIDFModel(SentSimModel):
    def __init__(self, args):
        super(TFIDFModel, self).__init__(args)
        self.index = None
        self.paths = dict(model=os.path.join(self.model_dir, 'tfidf.pkl'),
                          dict=os.path.join(self.model_dir, 'tfidf.dict'),
                          index=os.path.join(self.model_dir, 'tfidf.index'))

    def fit(self, list_toks):
        utils.verbose('Start training tfidf dictionary')
        self.dict = corpora.Dictionary(list_toks)

        utils.verbose('Start building tfidf corpus')
        self.corpus = [self.dict.doc2bow(toks) for toks in list_toks]

        utils.verbose('Start training tfidf model')
        self.model = models.TfidfModel(self.corpus)

        utils.verbose('Start saving tfidf dictionary and model')
        self.model.save(self.paths['model'])
        self.dict.save(self.paths['dict'])

        utils.verbose('Start building tfidf index')
        self.index = similarities.SparseMatrixSimilarity(self.model[self.corpus])
        # self.index = similarities.MatrixSimilarity(self.model[self.corpus])
        self.index.save(self.paths['index'])

    def get(self, toks):
        vec_bow = self.dict.doc2bow(toks)
        return self.model[vec_bow]

    def load(self):
        if all([os.path.exists(i) for i in self.paths.values()]):
            self.model = models.TfidfModel.load(self.paths['model'])
            utils.verbose('Load tfidf model from {}'.format(self.paths['model']))
            self.dict = corpora.Dictionary.load(self.paths['dict'])
            utils.verbose('Load tfidf dictionary from {}'.format(self.paths['dict']))
            self.index = similarities.SparseMatrixSimilarity.load(self.paths['index'])
            # self.index = similarities.MatrixSimilarity.load(self.paths['index'])
            utils.verbose('Load tfidf index from {}'.format(self.paths['index']))
        else:
            raise ValueError('Files under directory {} disappear'.format(self.model_dir))

    def search(self, toks, num):
        vec = self.get(toks)
        candidates = self.index[vec]
        candidates_sort = sorted(
            list(enumerate(candidates)), key=lambda item: item[1], reverse=True)
        return candidates_sort[: num]
