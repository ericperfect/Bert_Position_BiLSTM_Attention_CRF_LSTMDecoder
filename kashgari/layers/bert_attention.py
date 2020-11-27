#!/usr/bin/env python
#-*-coding:utf-8-*-

import kashgari
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import backend as K

L = keras.layers
initializers = keras.initializers
regularizers = keras.regularizers
constraints = keras.constraints
#np.random.seed(2018)

class bert_attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):

        self.nums = 4

        super(bert_attention, self).__init__(**kwargs)

    def build(self, input_shape):
        super(bert_attention, self).build(input_shape)


    def call(self, x):
        temp = x[:,:,0:768]
        for i in range(1,self.nums):
            temp = temp + x[:,:,768*i:768*(i+1)]

        return temp//4

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return (input_shape[0], input_shape[1], input_shape[2]//4)
kashgari.custom_objects['bert_attention'] = bert_attention