# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from annoy import AnnoyIndex

from src import utils
from src.traditional.keyword import load_keywords
from src.traditional.model import LDAModel, TFIDFModel


class Searcher(object):
    def __init__(self, args):
        self.tokenizer = args.tokenizer(args.path['vocab'])

    def search_line(self, line, num=15):
        raise NotImplementedError()


class DualEncoderSearcher(Searcher):
    def __init__(self, args):
        super(DualEncoderSearcher, self).__init__(args)
        self.infer_batch = args.batch(self.tokenizer, args.max_lens)
        self.model = args.model(args)
        self.ann = AnnoyIndex(args.hidden)
        self.ann.load(args.path['ann'])
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, args.path['model'])

    def search_line(self, line, num=15):
        input_x = self.infer_batch.encode_x(line)
        infer_features = {'input_x_ph': [input_x], 'keep_prob_ph': 1.0}
        infer_fetches, infer_feed = self.model.infer_step(infer_features)
        vec = self.sess.run(infer_fetches, infer_feed)[0][0]
        ids = self.ann.get_nns_by_vector(vec, num)
        vecs = [self.ann.get_item_vector(i) for i in ids]
        sim = [utils.cosine_similarity(vec, i) for i in vecs]
        return list(zip(ids, sim))


class LDASearcher(Searcher):
    def __init__(self, args):
        super(LDASearcher, self).__init__(args)
        self.model = LDAModel(args)
        self.model.load()

    def search_line(self, line, num=15):
        toks = [str(s) for s in self.tokenizer.encode_line_into_words(line)]
        return self.model.search(toks, num)


class TFIDFSearcher(Searcher):
    def __init__(self, args):
        super(TFIDFSearcher, self).__init__(args)
        self.model = TFIDFModel(args)
        self.keywords = load_keywords(args.path['model'])
        self.num_keywords = args.num_keywords
        self.model.load()

    def search_line(self, line, num=15):
        toks = [str(s) for s in self.tokenizer.encode_line_into_words(line)
                if s in self.keywords[: self.num_keywords]]
        return self.model.search(toks, num)
