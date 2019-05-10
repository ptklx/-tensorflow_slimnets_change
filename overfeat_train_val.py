#coding=utf-8

import tensorflow as tf 
import numpy as np 
import pdb
import os
import sys
sys.path.append('./slim')
from datetime import datetime
import slim.nets.overfeat as overfeat
from create_tf_record import *
import tensorflow.contrib.slim as slim

class OverFeatTest(tf.test.TestCase):
    def testTrainEvalWithReuse(self):
        train_batch_size = 2
        eval_batch_size = 1
        train_height, train_width = 231, 231
        eval_height, eval_width = 281, 281
        num_classes = 1000
        with self.test_session():
            train_inputs = tf.random_uniform(
                (train_batch_size, train_height, train_width, 3))
            logits, _ = overfeat.overfeat(train_inputs)
            self.assertListEqual(logits.get_shape().as_list(),
                                [train_batch_size, num_classes])



            tf.get_variable_scope().reuse_variables()
            eval_inputs = tf.random_uniform(
                (eval_batch_size, eval_height, eval_width, 3))
            logits, _ = overfeat.overfeat(eval_inputs, is_training=False,
                                            spatial_squeeze=False)
            self.assertListEqual(logits.get_shape().as_list(),
                                [eval_batch_size, 2, 2, num_classes])
            logits = tf.reduce_mean(logits, [1, 2])
            predictions = tf.argmax(logits, 1)
            self.assertEquals(predictions.get_shape().as_list(), [eval_batch_size])


if __name__ == "__main__":
    tf.test.main()






