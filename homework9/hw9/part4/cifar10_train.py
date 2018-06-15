# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.
Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.
Speed: With batch_size 128.
System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.
http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import cifar10
import math
import numpy as np
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 25000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")


def evaluate_set(sess, top_k_op, num_examples):
    num_iter = int(math.ceil(num_examples / FLAGS.batch_size))
    true_count = 0  # Counts the number of correct predictions.
    total_sample_count = num_iter * FLAGS.batch_size

    for step in range(num_iter):
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)

    # Compute precision
    return true_count / total_sample_count


def train():
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default():
        with tf.variable_scope("model") as scope:
            global_step = tf.Variable(0, trainable=False)

            # Get images and labels for CIFAR-10.
            # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
            # GPU and resulting in a slow down.
            with tf.device('/cpu:0'):
                images_train, labels_train = cifar10.distorted_inputs()
                images_test, labels_test = cifar10.inputs(eval_data=True)

            # Build a Graph that computes the logits predictions from the
            # inference model.
            logits_train = cifar10.inference(images_train)
            # Calculate loss.
            loss = cifar10.loss(logits_train, labels_train)
            # Build a Graph that trains the model with one batch of examples and
            # updates the model parameters.
            train_op = cifar10.train(loss, global_step)
        with tf.variable_scope("model", reuse=True):
            logits_test = cifar10.inference(images_test)

            # For evaluation
            top_k = tf.nn.in_top_k(logits_train, labels_train, 1)
            top_k_test = tf.nn.in_top_k(logits_test, labels_test, 1)

            summary_train_prec = tf.placeholder(tf.float32)  # summary writer for training data
            summary_test_prec = tf.placeholder(tf.float32)  # summary writer for testing data
            tf.summary.scalar('accuracy/train', summary_train_prec)  # train accuracy
            tf.summary.scalar('accuracy/test', summary_test_prec)  # test accuracy

            model_saver = tf.train.Saver(tf.all_variables())  # save the model by creating checkpoint
            summary_op = tf.summary.merge_all()  # merge all the summaries
            init = tf.initialize_all_variables()  # init the variables

            # Start running operations on the Graph.
            sess = tf.Session(config=tf.ConfigProto(
                log_device_placement=FLAGS.log_device_placement))
            sess.run(init)

            # Start the queue runners.
            tf.train.start_queue_runners(sess=sess)
            train_summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

            for step in range(FLAGS.max_steps):  # iterate through no of steps
                start_time = time.time()
                _, loss_value = sess.run([train_op, loss])
                duration = time.time() - start_time
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                # output the step loss after 10 batches
                if step % 10 == 0:
                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), step, loss_value,
                                        examples_per_sec, sec_per_batch))

                    prec_train = evaluate_set(sess, top_k, 1024)  # get train accuracy
                    prec_test = evaluate_set(sess, top_k_test, 1024)  # get test accuracy
                    print('%s: accuracy train = %.5f' % (datetime.now(), prec_train))
                    print('%s: accuracy test  = %.5f' % (datetime.now(), prec_test))
                    print("---------------------------------------------------------------------------------")

                # log the summary after every 100 steps
                if step % 100 == 0:
                    summary = sess.run(summary_op, feed_dict={summary_train_prec: prec_train,
                                                              summary_test_prec: prec_test})
                    train_summary_writer.add_summary(summary, step)  # create summary for testing and training accuracy

                # save the model after 1000 steps
                if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                    create_checkpoint = os.path.join(FLAGS.train_dir, 'model.ckpt')
                    model_saver.save(sess, create_checkpoint, global_step=step)


def main(argv=None):
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
