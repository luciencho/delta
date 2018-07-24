# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from src import utils
from src.tradictional.keyword import train_keywords
from src.tradictional.keyword import load_keywords
from src.tradictional.model import SentSimModel
from src.data_utils.vocab import Tokenizer


def process(args):
    utils.make_directory(args.path['model'])
    tokenizer = Tokenizer(args.path['vocab'])
    model = SentSimModel(args.path['model'], args.num_topics,
                         args.num_keywords, args.num_trees)
    train_x = utils.read_lines(args.path['train_x'])
    train_y = utils.read_lines(args.path['train_y'])
    dataset = train_x + train_y

    if args.problem == 'tfidf':
        trainset = [tokenizer.encode_line_into_words(i) for i in dataset]
        train_keywords(trainset, args.path['keyword'])
        keywords = load_keywords(args.path['keyword'])
        list_of_toks = []
        for n, line in enumerate(dataset):
            if not n % 10000 and n:
                utils.verbose('Tokenizing {} lines for {}'.format(n, args.problem))
            list_of_toks.append([str(s) for s in tokenizer.encode_line_into_words(line)
                                 if s in keywords[: args.num_keywords]])
        model.fit_tfidf(list_of_toks)
    elif args.problem == 'lda':
        list_of_toks = []
        for n, line in enumerate(dataset):
            if not n % 10000 and n:
                utils.verbose('Tokenizing {} lines for {}'.format(n, args.problem))
            list_of_toks.append([str(s) for s in tokenizer.encode_line_into_words(line)])
        model.fit_lda(list_of_toks)
    else:
        raise ValueError('Invalid problem: {}'.format(args.problem))
