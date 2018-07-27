# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function


import tensorflow as tf


_allowed_rnn_type = dict(
    lstm=tf.contrib.rnn.LSTMCell,
    gru=tf.contrib.rnn.GRUCell,
    rnn=tf.contrib.rnn.BasicRNNCell)


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
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

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
    """ Bidirectional RNN

    :param inputs: input tensor must be ranked 3 [batch_size, seq_len, emb_dim]
    :param seq_lens: input sequence lengths ranked 1 [batch_size]
    :param hidden: output size
    :param num_layers: number of layers
    :param rnn_type: rnn type
    :param keep_prob: (0, 1]
    :param scope: name scope
    :return:
        outputs: ([batch_size, seq_len, hidden], [batch_size, seq_len, hidden])
        states: [batch_size, hidden]
    """
    with tf.variable_scope(scope or 'bd_rnn'):
        fw_cell = rnn_cell(hidden, num_layers, rnn_type, keep_prob, 'fw_cell')
        bw_cell = rnn_cell(hidden, num_layers, rnn_type, keep_prob, 'bw_cell')
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell, inputs, seq_lens, dtype=tf.float32)
    if rnn_type.lower() == 'lstm':
        return outputs, states[-1][-1].h
    else:
        return outputs, states[-1][-1]


def conv_2d(embed_inputs, max_len, filter_sizes, num_filters, keep_prob):
    """ 2D CNN

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


def linear(inputs, output_size, use_bias, concat=False, scope=None):
    """ Linear layer

    :param inputs: A Tensor or a list of Tensors with shape [batch, input_size]
    :param output_size: An integer specify the output size
    :param use_bias: a boolean value indicate whether to use bias term
    :param concat: a boolean value indicate whether to concatenate all inputs
    :param scope: the scope of this layer, the default value is ``linear''
    :returns: a Tensor with shape [batch, output_size]
    :raises RuntimeError: raises ``RuntimeError'' when input sizes do not
                          compatible with each other
    """
    with tf.variable_scope(scope, default_name="linear", values=[inputs]):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        input_size = [item.get_shape()[-1].value for item in inputs]

        if len(inputs) != len(input_size):
            raise RuntimeError("inputs and input_size unmatched!")

        output_shape = tf.concat([tf.shape(inputs[0])[:-1], [output_size]],
                                 axis=0)
        # Flatten to 2D
        inputs = [tf.reshape(inp, [-1, inp.shape[-1].value]) for inp in inputs]

        results = []

        if concat:
            input_size = sum(input_size)
            inputs = tf.concat(inputs, 1)

            shape = [input_size, output_size]
            matrix = tf.get_variable("matrix", shape, dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer())
            results.append(tf.matmul(inputs, matrix))
        else:
            for i in range(len(input_size)):
                shape = [input_size[i], output_size]
                name = "matrix_%d" % i
                matrix = tf.get_variable(name, shape, dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer())
                results.append(tf.matmul(inputs[i], matrix))

        output = tf.add_n(results)

        if use_bias:
            shape = [output_size]
            bias = tf.get_variable("bias", shape, dtype=tf.float32)
            output = tf.nn.bias_add(output, bias)

        output = tf.reshape(output, output_shape)

        return output


def layer_norm(inputs, epsilon=1e-6, dtype=None, scope=None):
    """ Layer Normalization

    :param inputs: A Tensor of shape [..., channel_size]
    :param epsilon: A floating number
    :param dtype: An optional instance of tf.DType
    :param scope: An optional string
    :returns: A Tensor with the same shape as inputs
    """
    with tf.variable_scope(scope, default_name="layer_norm", values=[inputs],
                           dtype=dtype):
        channel_size = inputs.get_shape().as_list()[-1]

        scale = tf.get_variable("scale", shape=[channel_size],
                                initializer=tf.ones_initializer())

        offset = tf.get_variable("offset", shape=[channel_size],
                                 initializer=tf.zeros_initializer())

        mean = tf.reduce_mean(inputs, axis=-1, keep_dims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=-1,
                                  keep_dims=True)

        norm_inputs = (inputs - mean) * tf.rsqrt(variance + epsilon)

        return norm_inputs * scale + offset


def rcnn(inputs, seq_lens, rnn_hidden, hidden, keep_prob, scope=None):
    """ R-CNN

    :param inputs: embedding [batch_size, seq_len, emb_dim]
    :param seq_lens: sequence lengths for (bi)rnn
    :param rnn_hidden: hidden number for (bi)rnn
    :param hidden: hidden for text representation
    :param keep_prob: dropout keep probability
    :param scope: optional scope name
    :return: text level representation [batch_size, hidden]
    """
    with tf.variable_scope(scope or "rcnn"):
        (output_fw, output_bw), _ = bidirectional_rnn(
            inputs, seq_lens, rnn_hidden, 1, 'rnn', keep_prob, "rcnn_bd_lstm")

        with tf.name_scope("context"):
            shape = [tf.shape(output_fw)[0], 1, tf.shape(output_fw)[2]]
            con_l = tf.concat([tf.zeros(shape), output_fw[:, 1:]], axis=1, name="context_left")
            con_r = tf.concat([output_bw[:, :-1], tf.zeros(shape)], axis=1, name="context_right")

        with tf.name_scope("word_level"):
            x = tf.concat([con_l, inputs, con_r], axis=2)
            embedding_size = 2 * rnn_hidden + inputs.shape[-1].value

        with tf.name_scope("sentence_level"):
            w = tf.get_variable('w', [embedding_size, hidden], tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', [hidden], tf.float32,
                                initializer=tf.constant_initializer(0.1))
            out = tf.einsum('aij,jk->aik', x, w) + b
            out = tf.reduce_max(out, axis=1)
            out = layer_norm(tf.nn.softmax(out))

    return out
