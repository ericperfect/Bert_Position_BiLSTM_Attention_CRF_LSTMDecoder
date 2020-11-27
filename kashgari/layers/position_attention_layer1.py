#!/usr/bin/env python
#-*-coding:utf-8-*-
#windows

import kashgari
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import backend as K
import keras
L = keras.layers
#np.random.seed(2018)

class Position_attention_layer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Position_attention_layer, self).__init__(**kwargs)
        self.windows = 10
    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1].value+self.windows, input_shape[2].value),
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.W = self.add_weight(name='{}_W'.format(self.name),
                                 shape=(input_shape[-1].value,),  # 加value因为input_shape是TensorShape类型
                                 initializer='glorot_uniform',
                                 trainable=True)



        self.b = self.add_weight(name='{}_b'.format(self.name),
                                     shape=(input_shape[1].value+self.windows,),  # timesteps
                                     initializer='zero',
                                     trainable=True)


        super(Position_attention_layer, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        print(x)
        features_dim = x.shape[-1].value
        step_dim = x.shape[-2].value
        # print(K.reshape(self.kernel, (-1, features_dim)))  # n, d
        # print(K.reshape(self.W, (features_dim, 1)))  # w= dx1
        # print(K.dot(K.reshape(self.kernel, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))))  # nx1

        eij = K.reshape(K.dot(K.reshape(self.kernel, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))),
                        (-1, step_dim+self.windows))
        print(eij)

        eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)
        a = K.reshape(a,(step_dim+self.windows, 1))
        print(a)

        temp = a[0:self.windows,]
        print(temp)
        temp /= K.cast(K.sum(temp, axis=0, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = self.kernel[0:self.windows,] * temp
        alltemp = K.sum(weighted_input, axis=0, keepdims=True)

        for i in range(self.windows//2+1,step_dim+self.windows//2):
            temp = a[i-self.windows//2:i+self.windows//2,]
            temp /= K.cast(K.sum(temp, axis=0, keepdims=True) + K.epsilon(), K.floatx())
            weighted_input = self.kernel[i - self.windows//2:i + self.windows//2, ] * temp
            temp = K.sum(weighted_input, axis=0, keepdims=True)
            alltemp = keras.layers.concatenate([alltemp, temp], 0)

        print(alltemp)

        alltemp = keras.activations.tanh(alltemp)
        return x + alltemp

    def compute_output_shape(self, input_shape):
        return input_shape