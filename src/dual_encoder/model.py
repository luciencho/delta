# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from src.dual_encoder import common_layers


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
        feed_dict = {self.features[k[: -3]]: v for k, v in features.items()
                     if k.endswith('_ph')}
        fetches = [self.features[i] for i in fetch_names]
        return fetches, feed_dict

    def train_step(self, features):
        return self._steps(features, ['train_op', 'loss', 'acc'])

    def dev_step(self, features):
        return self._steps(features, ['loss', 'acc'])

    def infer_step(self, features):
        return self._steps(features, ['enc_x'])


class SoloModel(RetrievalModel):
    def __init__(self, hparam):
        super(SoloModel, self).__init__(hparam)
        self.features = self.bottom(self.features)
        self.features = self.embed_layer(self.features)
        self.features = self.encode_layer(self.features)
        self.features = self.interact_layer(self.features)
        self.features = self.top(self.features)

    def bottom(self, features):
        with tf.variable_scope('placeholders'):
            features['keep_prob'] = tf.placeholder(tf.float32, None, 'keep_prob')
            features['input_x'] = tf.placeholder(tf.int32, [None] * 3, 'input_x')
            features['input_y'] = tf.placeholder(tf.int32, [None] * 3, 'input_y')

        with tf.variable_scope('variables'):
            features['global_step'] = tf.Variable(0, trainable=False)
            features['x_lens'] = common_layers.length_last_axis(features['input_x'])
            features['y_lens'] = common_layers.length_last_axis(features['input_y'])
            features['labels'] = common_layers.get_labels(features['y_lens'])
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
            if self.hparam.encode_type == 'bidirectional_rnn':
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
            elif self.hparam.encode_type == 'rcnn':
                features['enc_x'] = common_layers.rcnn(
                    features['emb_x'],
                    features['x_lens'],
                    self.hparam.hidden,
                    self.hparam.hidden,
                    features['keep_prob'],
                    'text_representation')
                tf.get_variable_scope().reuse_variables()
                features['enc_y'] = common_layers.rcnn(
                    features['emb_y'],
                    features['y_lens'],
                    self.hparam.hidden,
                    self.hparam.hidden,
                    features['keep_prob'],
                    'text_representation')
            else:
                raise ValueError('Invalid encode_type: {}'.format(
                    self.hparam.encode_type))
        return features

    def interact_layer(self, features):
        with tf.variable_scope('interact_layer'):
            shape = [features['enc_y'].shape[-1].value, features['enc_x'].shape[-1].value]
            transformed_enc_x = features['enc_x']
            features['matrix'] = tf.get_variable(
                'matrix', shape, dtype=tf.float32,
                initializer=tf.truncated_normal_initializer())
            transformed_enc_y = tf.matmul(features['enc_y'], features['matrix'])
            if self.hparam.use_layer_norm:
                transformed_enc_x = common_layers.layer_norm(transformed_enc_x)
                transformed_enc_y = common_layers.layer_norm(transformed_enc_y)
            features['logits'] = tf.matmul(
                transformed_enc_y, transformed_enc_x, transpose_b=True)
        return features

    def top(self, features):
        with tf.variable_scope('top'):
            features['acc'] = tf.contrib.metrics.accuracy(
                predictions=tf.argmax(features['logits'], axis=-1),
                labels=tf.argmax(features['labels'], axis=-1))

            features['losses'] = tf.losses.softmax_cross_entropy(
                features['labels'], features['logits'])
            features['loss'] = tf.reduce_mean(features['losses'], name='show_loss')
            trainable_vars = tf.trainable_variables()
            features['extra_loss'] = tf.reduce_mean(
                self.hparam.l2_weight * tf.add_n(
                    [tf.nn.l2_loss(v) for v in trainable_vars if 'bias' not in v.name]),
                name='mean_loss')
            features['extra_loss'] += tf.losses.softmax_cross_entropy(
                tf.matmul(features['matrix'], features['matrix'], transpose_b=True),
                tf.eye(features['matrix'].shape[-1].value))

            features['learning_rate'] = tf.train.exponential_decay(
                self.hparam.learning_rate, features['global_step'],
                100, self.hparam.decay_rate)
            opt = tf.contrib.opt.LazyAdamOptimizer(learning_rate=features['learning_rate'])
            grads_vars = opt.compute_gradients(features['loss'] + features['extra_loss'])
            capped_grads_vars = [
                [tf.clip_by_value(g, - self.hparam.max_clip, self.hparam.max_clip),
                 v] for g, v in grads_vars if g is not None]
            features['train_op'] = opt.apply_gradients(capped_grads_vars, features['global_step'])
        return features


class PentaModel(RetrievalModel):
    def __init__(self, hparam):
        super(PentaModel, self).__init__(hparam)
        self.features = self.bottom(self.features)
        self.features = self.embed_layer(self.features)
        self.features = self.encode_layer(self.features)
        self.features = self.interact_layer(self.features)
        self.features = self.top(self.features)

    def bottom(self, features):
        with tf.variable_scope('placeholders'):
            features['keep_prob'] = tf.placeholder(tf.float32, None, 'keep_prob')
            features['input_x'] = tf.placeholder(tf.int32, [None] * 4, 'input_x')
            features['input_y'] = tf.placeholder(tf.int32, [None] * 3, 'input_y')

        with tf.variable_scope('variables'):
            features['global_step'] = tf.Variable(0, trainable=False)
            input_x = tf.split(features['input_x'], [1] * 5, 1)
            for i in range(1, 6):
                features['input_x_{}'.format(i)] = tf.squeeze(input_x[i - 1], 1)
                features['x_{}_lens'.format(i)] = common_layers.length_last_axis(
                    features['input_x_{}'.format(i)])
            features['y_lens'] = common_layers.length_last_axis(features['input_y'])
            features['labels'] = common_layers.get_labels(features['y_lens'])
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
            for i in range(1, 6):
                features['emb_x_{}'.format(i)] = common_layers.get_embedding(
                    features['word_vars'], features['char_vars'],
                    features['input_x_{}'.format(i)], features['keep_prob'])
            features['emb_y'] = common_layers.get_embedding(
                features['word_vars'], features['char_vars'],
                features['input_y'], features['keep_prob'])
        return features

    def encode_layer(self, features):
        with tf.variable_scope('encode_layer'):
            with tf.variable_scope('encode_each'):
                _, features['enc_y'] = common_layers.bidirectional_rnn(
                    features['emb_y'], features['y_lens'],
                    self.hparam.hidden, self.hparam.num_layers,
                    self.hparam.rnn_type, features['keep_prob'])
                tf.get_variable_scope().reuse_variables()
                enc_x = []
                for i in range(1, 6):
                    _, features['enc_x_{}'.format(i)] = common_layers.bidirectional_rnn(
                        features['emb_x_{}'.format(i)], features['x_{}_lens'.format(i)],
                        self.hparam.hidden, self.hparam.num_layers,
                        self.hparam.rnn_type, features['keep_prob'])
                    enc_x.append(tf.expand_dims(features['enc_x_{}'.format(i)], axis=0))
                enc_x = tf.transpose(tf.concat(enc_x, axis=0), [1, 0, 2])
            with tf.variable_scope('encode_all'):
                _, features['enc_x'] = common_layers.bidirectional_rnn(
                    enc_x, common_layers.length_last_axis(enc_x), self.hparam.hidden,
                    self.hparam.num_layers, self.hparam.rnn_type, features['keep_prob'])
        return features

    def interact_layer(self, features):
        with tf.variable_scope('interact_layer'):
            interact_hidden = features['enc_x'].shape[-1].value
            transformed_enc_x = features['enc_x']
            transformed_enc_y = common_layers.linear(
                features['enc_y'], interact_hidden, False)
            if self.hparam.use_layer_norm:
                transformed_enc_y = common_layers.layer_norm(transformed_enc_y)
            features['logits'] = tf.matmul(
                transformed_enc_y, transformed_enc_x, transpose_b=True)
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
