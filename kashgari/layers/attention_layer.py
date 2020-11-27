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

class Attention(tf.keras.layers.Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=False, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:

        """
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias

        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.step_dim = input_shape[1]
        assert len(input_shape) == 3 # batch ,timestep , num_features
        print(type(input_shape))
        self.W = self.add_weight(name='{}_W'.format(self.name),
                                 shape=(input_shape[-1].value,), #加value因为input_shape是TensorShape类型
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(name='{}_b'.format(self.name),
                                     shape=(input_shape[1].value,),#timesteps
                                     initializer='zero',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        super(Attention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        print(K.reshape(x, (-1, features_dim)))# n, d
        print(K.reshape(self.W, (features_dim, 1)))# w= dx1
        print(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))))#nx1

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))#batch,step
        print(eij)
        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        print(a)
        a = K.expand_dims(a)
        print("expand_dims:")
        print(a)
        print("x:")
        print(x)
        weighted_input = x * a #自动填充为相同的维度相乘 N T K
        print(weighted_input.shape)
        temp = K.sum(weighted_input, axis=1)  #N K
        temp = K.tile(K.expand_dims(temp, 1),[1,step_dim,1])
        temp = keras.layers.concatenate([x,temp])
        return temp

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return (input_shape[0], self.step_dim, 2*self.features_dim)
kashgari.custom_objects['Attention'] = Attention