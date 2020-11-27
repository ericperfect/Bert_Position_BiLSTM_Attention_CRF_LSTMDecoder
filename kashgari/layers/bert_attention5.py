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
        self.nums = 5
        super(bert_attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.features_dim = input_shape[-1]//self.nums
        self.step_dim = input_shape[1]
        self.W = self.add_weight(name='{}_W'.format(self.name),
                                 shape=(self.features_dim.value,),  # 加value因为input_shape是TensorShape类型
                                 initializer='glorot_uniform',
                                 trainable=True)

        self.b = self.add_weight(name='{}_b'.format(self.name),
                                 shape=(input_shape[1].value,),  # timesteps
                                 initializer='zero',
                                 )
        super(bert_attention, self).build(input_shape)


    def call(self, x):
        eij1 = K.reshape(
            K.dot(K.reshape(x[:, :, 0:768], (-1, self.features_dim)), K.reshape(self.W, (self.features_dim, 1))),
            (-1, self.step_dim))
        eij1 += self.b
        eij1 = K.expand_dims(eij1)

        eij2 = K.reshape(
            K.dot(K.reshape(x[:, :, 768:768*2], (-1, self.features_dim)), K.reshape(self.W, (self.features_dim, 1))),
            (-1, self.step_dim))
        eij2 += self.b
        eij2 = K.expand_dims(eij2)

        eij3 = K.reshape(
            K.dot(K.reshape(x[:, :, 768*2:768*3], (-1, self.features_dim)), K.reshape(self.W, (self.features_dim, 1))),
            (-1, self.step_dim))
        eij3 += self.b
        eij3 = K.expand_dims(eij3)

        eij4 = K.reshape(
            K.dot(K.reshape(x[:, :, 768*3:768*4], (-1, self.features_dim)), K.reshape(self.W, (self.features_dim, 1))),
            (-1, self.step_dim))
        eij4 += self.b
        eij4 = K.expand_dims(eij4)

        eij5 = K.reshape(
            K.dot(K.reshape(x[:, :, 768 * 4:768 * 5], (-1, self.features_dim)),
                  K.reshape(self.W, (self.features_dim, 1))),
            (-1, self.step_dim))
        eij5 += self.b
        eij5 = K.expand_dims(eij5)

        eij = keras.layers.concatenate([eij1, eij2, eij3, eij4, eij5], axis=2)
        print(eij)
        eij = K.tanh(eij)
        a = K.exp(eij)
        a /= K.cast(K.sum(a, axis=2, keepdims=True) + K.epsilon(), K.floatx())
        print(a)
        temp = a[:,:,0:1] * x[:, :, 0:768] + a[:,:,1:2] * x[:, :, 768:768*2] + a[:,:,2:3] * x[:, :, 768*2:768*3] + a[:,:,3:4] * x[:, :, 768*3:768*4] + a[:,:,4:5] * x[:, :, 768*4:768*5]
        print(temp)

        return temp

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return (input_shape[0], input_shape[1], input_shape[2]//4)
kashgari.custom_objects['bert_attention'] = bert_attention