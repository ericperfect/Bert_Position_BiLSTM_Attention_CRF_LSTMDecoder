#!/usr/bin/env python
#-*-coding:utf-8-*-
#跑不动
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

        self.features_dim = input_shape[-1]


        super(Attention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        t1 = x[:, 0, :]
        t1 = K.expand_dims(t1, 1)
        # t1 = K.tile(t1, [1, step_dim, 1])
        print(t1)
        eij = K.batch_dot(x, t1,(2,2))  #(?,500,1)
        # eij = K.tile(eij, [1, 1, features_dim])
        print(eij)
        a = K.exp(eij)
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        print(a)
        weighted_input = x * a
        temp = K.sum(weighted_input, axis=1)
        temp = K.expand_dims(temp, 1)
        temp = K.tile(temp, [1, 1, features_dim])
        print(temp)
        alltemp = temp

        for i in range(1,step_dim):
            t1 = x[:, i, :]
            t1 = K.expand_dims(t1, 1)
            # t1 = K.tile(t1, [1, 2, 1])
            eij = K.batch_dot(x, t1, (2,2))
            # eij = K.tile(eij, [1, 1, features_dim])
            a = K.exp(eij)
            a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
            weighted_input = x * a
            temp = K.sum(weighted_input, axis=1)
            temp = K.expand_dims(temp, 1)
            temp = K.tile(temp, [1, 1, features_dim])
            alltemp = keras.layers.concatenate([alltemp,temp],1)

        temp = keras.layers.concatenate([x,alltemp])
        return temp

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return (input_shape[0], self.step_dim, 2*self.features_dim)
kashgari.custom_objects['Attention'] = Attention