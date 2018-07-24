# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import time
from annoy import AnnoyIndex

from src import utils


def build_ann(args):
    vectors = []
    tokenizer = args.tokenizer(args.path['vocab'])
    infer_batch = args.batch(tokenizer, args.args.max_lens)
    infer_batch.set_data(utils.read_lines(args.path['train_x']),
                         utils.read_lines(args.path['train_y']))
    model = args.model(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, args.path['model'])
        starter = time.time()
        idx = 0
        update_epoch = False
        i = 0
        while not update_epoch:
            input_x, input_y, idx, update_epoch = infer_batch.next_batch(
                args.batch_size, idx)
            infer_features = {'input_x_ph': input_x, 'keep_prob_ph': 1.0}
            infer_fetches, infer_feed = model.infer_step(infer_features)
            enc_questions = sess.run(infer_fetches, infer_feed)
            vectors += enc_questions
            if i % args.show_steps == 0 and i:
                speed = args.show_steps / (time.time() - starter)
                utils.verbose('step : {:05d} | speed: {:.5f} it/s'.format(i, speed))
                starter = time.time()
            i += 1
    vectors = np.reshape(np.array(vectors), [-1, args.hidden])[: infer_batch.data_size]
    vec_dim = vectors.shape[-1]
    ann = AnnoyIndex(vec_dim)
    for n, ii in enumerate(vectors):
        ann.add_item(n, ii)
    ann.build(10)
    return ann


def process(args):
    ann = build_ann(args)
    ann.save(args.path['ann'])
    utils.verbose('dump annoy into {}'.format(args.path['ann']))
