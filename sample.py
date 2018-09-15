#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
from six.moves import cPickle


from six import text_type

import tensorflow as tf
from model import Model


def sample(args, prime):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    # Use most frequent char if no prime is given
    if prime == '':
        prime = chars[0]
    model = Model(saved_args, training=False)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            with open('output/output.txt', 'w') as f:
                f.write(str(model.sample(sess, chars, vocab, args.n, prime,
                                         args.sample).encode('utf-8')))

# must be called from python file


if __name__ == '__main__':
    # sample(args)
    print("Sample Initialized")
