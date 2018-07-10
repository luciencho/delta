# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function


import tensorflow as tf


_allowed_rnn_type = dict(
    lstm=tf.contrib.rnn.LSTMCell,
    gru=tf.contrib.rnn.GRUCell,
    rnn=tf.contrib.rnn.RNNCell)


def length_last_axis(tensor):
    lens = tf.reduce_sum(tf.sign(tensor), axis=1)
    if len(lens.shape) == 2:
        lens = tf.reduce_mean(lens, axis=-1)
    return tf.cast(lens, dtype=tf.int32)


def rnn_attention(inputs, attention_size, return_alphas, name_scope=None):
    with tf.variable_scope('rnn_attention' or name_scope):
        hidden_size = inputs.shape[-1].value

        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        with tf.name_scope('v'):
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        if not return_alphas:
            return output
        else:
            return output, alphas


def get_labels(tensor):
    return tf.matrix_diag(tf.sign(tensor))


def rnn_cell(hidden, num_layers=1, rnn_type='lstm', dropout=0.8, scope=None):

    def create_rnn_cell():
        cell = _allowed_rnn_type.get(rnn_type.lower(), 'rnn')(hidden, reuse=reuse)
        return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)

    with tf.variable_scope(scope or 'rnn'):
        reuse = None if not tf.get_variable_scope().reuse else True
        return tf.contrib.rnn.MultiRNNCell(
            [create_rnn_cell() for _ in range(num_layers)], state_is_tuple=True)


def bidirectional_rnn(inputs, seq_lens, hidden, num_layers=1,
                      rnn_type='lstm', keep_prob=0.8, scope=None):
    """

    :param inputs: input tensor must be ranked 3 [batch_size, seq_len, emb_dim]
    :param seq_lens: input sequence lengths ranked 1 [batch_size]
    :param hidden: output size
    :param num_layers: number of layers
    :param rnn_type: rnn type
    :param keep_prob: (0, 1]
    :param scope: name scope
    :return:
        outputs: [batch_size, seq_len, 2 * hidden]
        states: [batch_size, hidden]
    """
    with tf.variable_scope(scope or 'bd_rnn'):
        fw_cell = rnn_cell(hidden, num_layers, rnn_type, keep_prob, 'fw_cell')
        bw_cell = rnn_cell(hidden, num_layers, rnn_type, keep_prob, 'bw_cell')
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell, inputs, seq_lens, dtype=tf.float32)
    if rnn_type.lower() == 'lstm':
        return tf.concat(outputs, axis=-1), states[-1][-1].h
    elif rnn_type.lower() == 'gru':
        return tf.concat(outputs, axis=-1), states[-1][-1]


def conv_2d(embed_inputs, max_len, filter_sizes, num_filters, keep_prob):
    """

    :param embed_inputs: input tensor must be ranked 3 [batch_size, seq_len, emb_dim]
    :param max_len: maximal length for inputs
    :param filter_sizes: something like [2, 3, 4]
    :param num_filters: number of filters
    :param keep_prob: (0, 1]
    :return:
        output [batch_size, len(filter_sizes) * num_filters]
    """
    emb_dim = embed_inputs.shape[-1].value
    embed_inputs = tf.expand_dims(embed_inputs, -1)
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            filter_shape = [filter_size, emb_dim, 1, num_filters]
            w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="w")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(
                embed_inputs,
                w,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, max_len - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            pooled_outputs.append(pooled)

    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    output = tf.nn.dropout(h_pool_flat, keep_prob)
    return output


def get_embedding(word_vars, char_vars, inputs, keep_prob):
    word_emb, char_emb = tf.split(inputs, 2, axis=-1)
    word_emb = tf.squeeze(word_emb, -1)
    char_emb = tf.squeeze(char_emb, -1)
    with tf.device('/cpu:0'):
        emb_words = tf.nn.embedding_lookup(word_vars, word_emb)
        emb_chars = tf.nn.embedding_lookup(char_vars, char_emb)
    emb_outputs = tf.nn.dropout(tf.concat([emb_words, emb_chars], axis=-1), keep_prob)
    return emb_outputs
