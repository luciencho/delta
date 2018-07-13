# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os
import re
import jieba_fast as jieba

from src import utils


jieba.initialize()
PAD, UNK, EOS = copy_head = ['<pad>', '<unk>', '<eos>']


allowed_suffix = [
    'com', 'net', 'org', 'gov', 'mil', 'edu', 'biz', 'info', 'pro', 'name', 'coop',
    'travel', 'xxx', 'idv', 'aero', 'museum', 'mobi', 'asia', 'tel', 'int', 'post',
    'jobs', 'cat', 'ac', 'ad', 'ae', 'af', 'ag', 'ai', 'al', 'am', 'an', 'ao', 'aq',
    'ar', 'as', 'at', 'au', 'aw', 'az', 'ba', 'bb', 'bd', 'be', 'bf', 'bg', 'bh', 'bi',
    'bj', 'bm', 'bn', 'bo', 'br', 'bs', 'bt', 'bv', 'bw', 'by', 'bz', 'ca', 'cc', 'cd',
    'cf', 'cg', 'ch', 'ci', 'ck', 'cl', 'cm', 'cn', 'co', 'cr', 'cu', 'cv', 'cx', 'cy',
    'cz', 'de', 'dj', 'dk', 'dm', 'do', 'dz', 'ec', 'ee', 'eg', 'eh', 'er', 'es', 'et',
    'eu', 'fi', 'fj', 'fk', 'fm', 'fo', 'fr', 'ga', 'gd', 'ge', 'gf', 'gg', 'gh', 'gi',
    'gl', 'gm', 'gn', 'gp', 'gq', 'gr', 'gs', 'gt', 'gu', 'gw', 'gy', 'hk', 'hm', 'hn',
    'hr', 'ht', 'hu', 'id', 'ie', 'il', 'im', 'in', 'io', 'iq', 'ir', 'is', 'it', 'je',
    'jm', 'jo', 'jp', 'ke', 'kg', 'kh', 'ki', 'km', 'kn', 'kp', 'kr', 'kw', 'ky', 'kz',
    'la', 'lb', 'lc', 'li', 'lk', 'lr', 'ls', 'ma', 'mc', 'md', 'me', 'mg', 'mh', 'mk',
    'ml', 'mm', 'mn', 'mo', 'mp', 'mq', 'mr', 'ms', 'mt', 'mu', 'mv', 'mw', 'mx', 'my',
    'mz', 'na', 'nc', 'ne', 'nf', 'ng', 'ni', 'nl', 'no', 'np', 'nr', 'nu', 'nz', 'om',
    'pa', 'pe', 'pf', 'pg', 'ph', 'pk', 'pl', 'pm', 'pn', 'pr', 'ps', 'pt', 'pw', 'py',
    'qa', 're', 'ro', 'ru', 'rw', 'sa', 'sb', 'sc', 'sd', 'se', 'sg', 'sh', 'si', 'sj',
    'sk', 'sm', 'sn', 'so', 'sr', 'st', 'sv', 'sy', 'sz', 'tc', 'td', 'tf', 'tg', 'th',
    'tj', 'tk', 'tl', 'tm', 'tn', 'to', 'tp', 'tr', 'tt', 'tv', 'tw', 'tz', 'ua', 'ug',
    'uk', 'um', 'us', 'uy', 'uz', 'va', 'vc', 've', 'vg', 'vi', 'vn', 'vu', 'wf', 'ws',
    'ye', 'yt', 'yu', 'yr', 'za', 'zm', 'zw']


def strip_line(line):
    return line.strip()


def del_space(line):
    return re.sub(r' +', ' ', line)


def sub_email(line):
    url_re = re.compile(
        r'(([a-zA-Z0-9_-]+\.)*[a-zA-Z0-9_-]+@([a-zA-Z0-9-]+\.)+[a-zA-Z]+)(?![a-zA-Z])')
    new_line = line
    resu = []
    for i in url_re.finditer(line):
        curr = line[i.start(): i.end()]
        for a in allowed_suffix:
            if curr.endswith('.' + a):
                resu.append(line[i.start(): i.end()])
    for r in resu:
        new_line = re.sub(r, r'[邮箱x]', new_line)
    return new_line


def sub_url(line):
    url_re = re.compile(
        r'(([a-zA-Z]+:\\\\|[a-zA-Z]+://)?([a-zA-Z0-9-]+\.)'
        r'+[a-zA-Z]{2,10}[a-zA-Z0-9-_ ./?#%&=]*)(?![a-zA-Z])|'
        r'(([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,10})(?![a-zA-Z])')
    new_line = line
    resu = []
    for i in url_re.finditer(line):
        curr = line[i.start(): i.end()]
        for a in allowed_suffix:
            if curr.endswith('.' + a):
                resu.append(line[i.start(): i.end()])
    for r in resu:
        new_line = re.sub(r, r'[链接x]', new_line)
    return new_line


default_regex = [
    (re.compile(r'\[ORDERID_\d+\]'), 'order_id'),
    (re.compile(r'#E-s\d*\[数字x\]'), 'emoji'),
    (re.compile(r'\[邮箱x\]'), 'email'),
    (re.compile(r'\[数字x\]'), 'number'),
    (re.compile(r'\[地址x\]'), 'location'),
    (re.compile(r'\[时间x\]'), 'time'),
    (re.compile(r'\[日期x\]'), 'date'),
    (re.compile(r'\[链接x\]'), 'link'),
    (re.compile(r'\[电话x\]'), 'phone'),
    (re.compile(r'\[金额x\]'), 'price'),
    (re.compile(r'\[姓名x\]'), 'name'),
    (re.compile(r'\[站点x\]'), 'station'),
    (re.compile(r'\[身份证号x\]'), 'photo_id'),
    (re.compile(r'\[组织机构x\]'), 'organization'),
    (re.compile(r'\[子\]'), 'non-sense'),
    (re.compile(r'\[父原始\]'), 'non-sense'),
    (re.compile(r'\[父\]'), 'non-sense'),
    (re.compile(r'~O\(∩_∩\)O/~'), 'smiler'),
    (re.compile(r'<s>'), 'splitter'),
    (re.compile(r'\d{11}'), 'phone'),
    (re.compile(r'\d{6,10}'), 'number'),
    (re.compile(r'\d{11,15}'), 'number'),
    (re.compile(r'&nbsp;'), 'non-sense')
]


default_clean_fns = [del_space, sub_email, sub_url, strip_line]


class SubCutter(object):
    def __init__(self):
        self.collector = None
        self.counter = None

    def _reset(self):
        self.collector = []
        self.counter = 0

    def _sub_fn(self, tag, with_tag):

        def sub_with_num(matched):
            if with_tag:
                self.collector.append('{{{{{}:{}}}}}'.format(tag, matched.group()))
            else:
                self.collector.append(matched.group())
            self.counter += 1
            return 'ph_{}_{}_'.format(tag, self.counter)

        return sub_with_num

    def _segment(self, line, with_tag=True):
        self._reset()
        new_line = line
        for r, ta in default_regex:
            new_line = r.sub(self._sub_fn(ta, with_tag), new_line)
        matched = re.finditer(r'ph_[\s\S]+?_(?P<num>\d{1,2})_', new_line)
        start = 0
        tokens = []
        for m in matched:
            tokens += jieba.lcut(new_line[start: m.start()])
            tokens += [self.collector[int(m.group('num')) - 1]]
            start = m.end()
        tokens += jieba.lcut(new_line[start:])
        return tokens

    def cut(self, line):
        new_line = line
        for fn in default_clean_fns:
            new_line = fn(new_line)
        return self._segment(new_line)


class Tokenizer(object):
    def __init__(self, files=None):
        """ Char-base adding word Tokenizer

        :param files: [word_file_path, char_file_path]
        """
        self.word_counter = {}
        self.char_counter = {}
        if files is not None:
            self.words = utils.read_lines(files[0])
            self.chars = utils.read_lines(files[1])
            utils.verbose('loading words from file {} with word size {}'.format(
                files[0], self.word_size))
            utils.verbose('loading chars from file {} with char size {}'.format(
                files[1], self.char_size))
        else:
            self.words = []
            self.chars = []
        self.cutter = SubCutter()
        self.word_dict = dict()
        self.char_dict = dict()
        self._set_dict()
        self.PAD_ID = 0
        self.UNK_ID = 1
        self.EOS_ID = 2

    def _set_dict(self):
        self.word_dict = {w: i for i, w in enumerate(self.words)}
        self.char_dict = {c: i for i, c in enumerate(self.chars)}

    @property
    def word_size(self):
        return len(self.words)

    @property
    def char_size(self):
        return len(self.chars)

    def _collect_vocab(self, lines):

        def insert(counter, obj):
            if obj in counter:
                counter[obj] += 1
            else:
                counter[obj] = 1
            return counter

        word_counter = dict()
        char_counter = dict()
        for n, line in enumerate(lines, start=1):
            if not n % 10000:
                utils.verbose('processing no.{} lines'.format(n))
            words = self.cutter.cut(line)
            for word in words:
                if word.startswith('{{') and word.endswith('}}'):
                    new_word = '<' + word.split(':')[0][2:] + '>'
                    word_counter = insert(word_counter, new_word)
                    char_counter = insert(char_counter, new_word)
                else:
                    word_counter = insert(word_counter, word)
                    for char in word:
                        char_counter = insert(char_counter, char)
        word_counter = sorted(word_counter, key=word_counter.get, reverse=True)
        char_counter = sorted(char_counter, key=char_counter.get, reverse=True)
        return word_counter, char_counter

    def _set_vocab(self, data, word_size, char_size):
        self.word_counter, self.char_counter = self._collect_vocab(data)
        self.words = copy_head + list(self.word_counter)[: word_size - len(copy_head)]
        self.chars = copy_head + list(self.char_counter)[: char_size - len(copy_head)]
        utils.verbose('real words: {}, final words: {}'.format(
            len(self.word_counter) + 3, len(self.words)))
        utils.verbose('real chars: {}, final chars: {}'.format(
            len(self.char_counter) + 3, len(self.chars)))
        self._set_dict()

    def build_vocab(self, data, token_limits, files):
        """ Build words and chars with limited sizes and write into files

        :param data: list of lines
        :param token_limits: word_limit_size, char_limit_size
        :param files: word_file_path, char_file_path
        :return:
        """
        self._set_vocab(data, token_limits[0], token_limits[1])
        utils.write_lines(files[0], self.words)
        utils.verbose(
            'words has been dumped in {}'.format(os.path.abspath(files[0])))
        utils.write_lines(files[1], self.chars)
        utils.verbose(
            'chars has been dumped in {}'.format(os.path.abspath(files[1])))

    def encode_line(self, line):
        """ Encode one line to list of word-char id pairs

        :param line: string
        :return: list of (word_id, char_id)
        """
        words = self.cutter.cut(line)
        wc_pairs = []
        for word in words:
            if word.startswith('{{') and word.endswith('}}'):
                word = '<' + word.split(':')[0][2:] + '>'
                word_id = self.word_dict.get(word, self.word_dict[UNK])
                char_id = self.char_dict.get(word, self.char_dict[UNK])
                wc_pairs.append((word_id, char_id))
            else:
                word_id = self.word_dict.get(word, self.word_dict[UNK])
                for char in word:
                    char_id = self.char_dict.get(char, self.char_dict[UNK])
                    wc_pairs.append((word_id, char_id))
        return wc_pairs

    def encode_line_trad(self, line):
        words = self.cutter.cut(line)
        tokens = []
        for word in words:
            if word.startswith('{{') and word.endswith('}}'):
                word = '<' + word.split(':')[0][2:] + '>'
            word_id = self.word_dict.get(word, self.word_dict[UNK])
            tokens.append(word_id)
        return tokens
