#!/usr/bin/env python
#-*-coding:utf-8-*-


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

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1].value, input_shape[2].value),
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.W = self.add_weight(name='{}_W'.format(self.name),
                                 shape=(input_shape[-1].value,),  # 加value因为input_shape是TensorShape类型
                                 initializer='glorot_uniform',
                                 trainable=True)

        self.W2 = self.add_weight(name='{}_W'.format(self.name),
                                 shape=(2*input_shape[-1].value,input_shape[-1].value),  # 加value因为input_shape是TensorShape类型
                                 initializer='glorot_uniform',
                                 trainable=True)


        self.b = self.add_weight(name='{}_b'.format(self.name),
                                     shape=(input_shape[1].value,),  # timesteps
                                     initializer='zero',
                                     trainable=True)

        self.b2 = self.add_weight(name='{}_b'.format(self.name),
                                 shape=(input_shape[-1].value,),  # timesteps
                                 initializer='zero',
                                 trainable=True)

        super(Position_attention_layer, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        print(x)
        features_dim = x.shape[-1].value
        step_dim = x.shape[-2].value
        print(K.reshape(self.kernel, (-1, features_dim)))  # n, d
        print(K.reshape(self.W, (features_dim, 1)))  # w= dx1
        print(K.dot(K.reshape(self.kernel, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))))  # nx1

        eij = K.reshape(K.dot(K.reshape(self.kernel, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))),
                        (-1, step_dim))  # batch,step
        print(eij)

        eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)


        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = tf.transpose(a,(1,0))
        print(a)

        print("x:")
        print(self.kernel)
        weighted_input = self.kernel * a  # 自动填充为相同的维度相乘 N T K
        print(weighted_input.shape)
        temp = K.sum(weighted_input, axis=0)  # N K  权重相加
        temp = K.tile(K.expand_dims(temp, 0), [step_dim, 1])
        temp = keras.layers.concatenate([self.kernel, temp])
        temp = K.dot(temp, self.W2) + self.b2
        return x + temp

    def compute_output_shape(self, input_shape):
        return input_shape