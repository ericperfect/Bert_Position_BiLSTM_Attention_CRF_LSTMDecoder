#!/usr/bin/env python
#-*-coding:utf-8-*-

import kashgari
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import backend as K

L = keras.layers
#np.random.seed(2018)

class Position_layer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Position_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1].value, input_shape[2].value),
                                      initializer='glorot_uniform',
                                      trainable=True)

        super(Position_layer, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        return x + self.kernel

    def compute_output_shape(self, input_shape):
        return input_shape