# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import time

from src import utils
from src.retrieval.model import DualEncoderModel
from src.data_utils.data import SoloBatch
from src.data_utils.vocab import Tokenizer


class Recorder(object):
    def __init__(self):
        self.lowest_loss = 10
        self.train_idx = 0
        self.dev_idx = 0
        self.train_losses = []
        self.dev_losses = []
        self.train_accs = []
        self.dev_accs = []

    def reset(self):
        self.train_losses = []
        self.dev_losses = []
        self.train_accs = []
        self.dev_accs = []

    def stats(self):
        train_loss = sum(self.train_losses) / len(self.train_losses)
        dev_loss = sum(self.dev_losses) / len(self.dev_losses)
        train_acc = sum(self.train_accs) / len(self.train_accs)
        dev_acc = sum(self.dev_accs) / len(self.dev_accs)
        self.reset()
        save = False
        if self.lowest_loss > dev_loss:
            save = True
            self.lowest_loss = dev_loss
        return {'train_loss': train_loss, 'dev_loss': dev_loss,
                'train_acc': train_acc, 'dev_acc': dev_acc, 'save': save}


def process(args):
    utils.clean_and_make_directory(args.path['model'])
    tokenizer = Tokenizer(args.path['vocab'])
    train_batch = SoloBatch(tokenizer, [args.x_max_len, args.y_max_len])
    train_batch.set_data(utils.read_lines(args.path['train_x']),
                         utils.read_lines(args.path['train_y']))
    dev_batch = SoloBatch(tokenizer, [args.x_max_len, args.y_max_len])
    dev_batch.set_data(utils.read_lines(args.path['dev_x']),
                       utils.read_lines(args.path['dev_y']))
    model = DualEncoderModel(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(pad_step_number=True)
        recorder = Recorder()
        starter = time.time()

        for i in range(args.max_steps):
            input_x, input_y, idx, update_epoch = train_batch.next_batch(
                args.batch_size, recorder.train_idx)
            train_features = {'input_x_ph': input_x,
                              'input_y_ph': input_y, 'keep_prob_ph': args.keep_prob}
            recorder.train_idx = idx
            train_fetches, train_feed = model.train_step(train_features)
            _, train_loss, train_acc = sess.run(train_fetches, train_feed)
            recorder.train_losses.append(train_loss)
            recorder.train_accs.append(train_acc)

            if not i % args.show_steps and i:
                input_x, input_y, idx, update_epoch = dev_batch.next_batch(
                    args.batch_size, recorder.dev_idx)
                dev_features = {'input_x_ph': input_x,
                                'input_y_ph': input_y, 'keep_prob_ph': 1.0}
                recorder.dev_idx = idx
                dev_fetches, dev_feed = model.dev_step(dev_features)
                dev_loss, dev_acc = sess.run(dev_fetches, dev_feed)
                recorder.dev_losses.append(dev_loss)
                recorder.dev_accs.append(dev_acc)
                speed = args.show_steps / (time.time() - starter)
                utils.verbose(
                    r'        step {:05d} | train [{:.5f} {:.5f}] | '
                    r'dev [{:.5f} {:.5f}] | speed {:.5f} it/s'.format(
                        i, train_loss, train_acc, dev_loss, dev_acc, speed))
                starter = time.time()

            if not i % args.save_steps and i:
                features = recorder.stats()
                if features['save']:
                    saver.save(sess, args.path['model'])
                utils.verbose(
                    r'step {:05d} - {:05d} | train [{:.5f} {:.5f}] | '
                    r'dev [{:.5f} {:.5f}]'.format(
                        i - args.save_steps, i, features['train_loss'],
                        features['train_acc'], features['dev_loss'], features['dev_acc']))
                print('-+' * 55)

    utils.write_result(args, recorder.lowest_loss, args.path['model'])
