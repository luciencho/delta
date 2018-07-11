# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from src.retrieval import common_layers


class RetrievalModel(object):
    def __init__(self, hparam):
        self.hparam = hparam
        self.features = {}

    def bottom(self, features):
        """ Construct placeholders and basic variables

        :param features:
        :return:
        """
        raise NotImplementedError()

    def embed_layer(self, features):
        """ Embed every sequence

        :param features:
        :return:
        """
        raise NotImplementedError()

    def encode_layer(self, features):
        """ Encode every embedded sequence

        :param features:
        :return:
        """
        raise NotImplementedError()

    def interact_layer(self, features):
        """ Interacting sequences and return calculate logits

        :param features:
        :return:
        """
        raise NotImplementedError

    def top(self, features):
        """ Extra operations like loss, acc, train op, etc.

        :param features:
        :return:
        """
        raise NotImplementedError()

    def _steps(self, features, fetch_names):
        feed_dict = {self.features.get(i[: -3]): features[i] for i in features
                     if i.endswith('_ph')}
        fetches = [self.features[i] for i in fetch_names]
        return fetches, feed_dict

    def train_step(self, features):
        return self._steps(features, ['train_op', 'loss', 'acc'])

    def dev_step(self, features):
        return self._steps(features, ['loss', 'acc'])

    def infer_step(self, features):
        return self._steps(features, ['enc_x'])


class DualEncoderModel(RetrievalModel):
    def __init__(self, hparam):
        super(DualEncoderModel, self).__init__(hparam)
        self.features = self.bottom(self.features)
        self.features = self.embed_layer(self.features)
        self.features = self.encode_layer(self.features)
        self.features = self.interact_layer(self.features)
        self.features = self.top(self.features)

    def bottom(self, features):
        with tf.variable_scope('placeholders'):
            features['keep_prob'] = tf.placeholder(tf.float32, None, 'keep_prob')
            features['input_x'] = tf.placeholder(tf.int32, [None, None, None], 'input_x')
            features['input_y'] = tf.placeholder(tf.int32, [None, None, None], 'input_y')

        with tf.variable_scope('variables'):
            features['global_step'] = tf.Variable(0, trainable=False)
            features['x_lens'] = common_layers.length_last_axis(features['input_x'])
            features['y_lens'] = common_layers.length_last_axis(features['input_y'])
            features['labels'] = common_layers.get_labels(features['x_lens'])
        return features

    def embed_layer(self, features):
        with tf.variable_scope('embed_layer'):
            with tf.device('/cpu:0'):
                features['word_vars'] = tf.get_variable(
                    'word_vars', [self.hparam.word_size, self.hparam.emb_dim],
                    initializer=tf.contrib.layers.xavier_initializer())
                features['char_vars'] = tf.get_variable(
                    'char_vars', [self.hparam.char_size, self.hparam.emb_dim],
                    initializer=tf.contrib.layers.xavier_initializer())
            features['emb_x'] = common_layers.get_embedding(
                features['word_vars'],
                features['char_vars'],
                features['input_x'],
                features['keep_prob'])
            features['emb_y'] = common_layers.get_embedding(
                features['word_vars'],
                features['char_vars'],
                features['input_y'],
                features['keep_prob'])
        return features

    def encode_layer(self, features):
        with tf.variable_scope('encode_layer'):
            _, features['enc_x'] = common_layers.bidirectional_rnn(
                features['emb_x'],
                features['x_lens'],
                self.hparam.hidden,
                self.hparam.num_layers,
                self.hparam.rnn_type,
                features['keep_prob'])
            tf.get_variable_scope().reuse_variables()
            _, features['enc_y'] = common_layers.bidirectional_rnn(
                features['emb_y'],
                features['y_lens'],
                self.hparam.hidden,
                self.hparam.num_layers,
                self.hparam.rnn_type,
                features['keep_prob'])
        return features

    def interact_layer(self, features):
        with tf.variable_scope('interact_layer'):
            transformed_enc_y = common_layers.linear(
                features['enc_y'], features['enc_x'].shape[-1], True)
            if self.hparam.use_layer_norm:
                transformed_enc_y = common_layers.layer_norm(transformed_enc_y)
            features['logits'] = tf.matmul(
                transformed_enc_y, features['enc_x'], transpose_b=True)
        return features

    def top(self, features):
        with tf.variable_scope('top'):
            features['losses'] = tf.losses.softmax_cross_entropy(
                features['labels'], features['logits'])
            features['acc'] = tf.contrib.metrics.accuracy(
                predictions=tf.argmax(features['logits'], axis=-1),
                labels=tf.argmax(features['labels'], axis=-1))
            features['loss'] = tf.reduce_mean(features['losses'], name='show_loss')
            trainable_vars = tf.trainable_variables()
            features['extra_loss'] = tf.reduce_mean(
                self.hparam.l2_weight * tf.add_n(
                    [tf.nn.l2_loss(v) for v in trainable_vars if 'bias' not in v.name]),
                name='mean_loss')

            features['learning_rate'] = tf.train.exponential_decay(
                self.hparam.learning_rate, features['global_step'],
                100, self.hparam.decay_rate)
            opt = tf.contrib.opt.LazyAdamOptimizer(learning_rate=features['learning_rate'])
            grads_vars = opt.compute_gradients(features['loss'] + features['extra_loss'])
            capped_grads_vars = [[
                tf.clip_by_value(g, -1, 1), v] for g, v in grads_vars if g is not None]
            features['train_op'] = opt.apply_gradients(capped_grads_vars, features['global_step'])
        return features


def solo_lstm():  # 3.114 30.39%
    hparams = tf.contrib.training.HParams(
        word_size=50000,
        char_size=5000,
        emb_dim=256,
        hidden=128,
        num_layers=1,
        rnn_type='lstm',
        use_layer_norm=False,
        l2_weight=1e-5,
        learning_rate=5e-3,
        decay_rate=0.98,
        keep_prob=0.7,
        max_steps=20000,
        show_steps=50,
        save_steps=250,
        batch_size=256,
        x_max_len=128,
        y_max_len=64)
    return hparams


def solo_gru():
    hparams = solo_lstm()
    hparams.rnn_type = 'gru'
    return hparams


def solo_lstm_v1():
    hparams = solo_lstm()
    return hparams


def solo_lstm_ln():
    hparams = solo_lstm()
    hparams.use_layer_norm = True
    return hparams
