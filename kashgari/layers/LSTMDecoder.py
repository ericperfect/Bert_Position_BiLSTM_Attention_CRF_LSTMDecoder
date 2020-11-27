#!/usr/bin/env python
#-*-coding:utf-8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras
import collections

import numpy as np
import kashgari
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras.layers.recurrent import PeepholeLSTMCell
class LSTMCellDecoder(PeepholeLSTMCell):

    """Equivalent to LSTMCell class but adds peephole connections.

    Peephole connections allow the gates to utilize the previous internal state as
    well as the previous hidden state (which is what LSTMCell is limited to).
    This allows PeepholeLSTMCell to better learn precise timings over LSTMCell.

    From [Gers et al.](http://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf):

    "We find that LSTM augmented by 'peephole connections' from its internal
    cells to its multiplicative gates can learn the fine distinction between
    sequences of spikes spaced either 50 or 49 time steps apart without the help
    of any short training exemplars."

    The peephole implementation is based on:

    [Long short-term memory recurrent neural network architectures for
     large scale acoustic modeling.
    ](https://research.google.com/pubs/archive/43905.pdf)

    Example:

    ```python
    # Create 2 PeepholeLSTMCells
    peephole_lstm_cells = [PeepholeLSTMCell(size) for size in [128, 256]]
    # Create a layer composed sequentially of the peephole LSTM cells.
    layer = RNN(peephole_lstm_cells)
    input = keras.Input((timesteps, input_dim))
    output = layer(input)
    ```
    """

    def build(self, input_shape):
        super(LSTMCellDecoder, self).build(input_shape)
        # The following are the weight matrices for the peephole connections. These
        # are multiplied with the previous internal state during the computation of
        # carry and output.

        self.z_peephole_weights = self.add_weight(
            shape=(self.units,),
            name='z_peephole_weights',
            initializer=self.kernel_initializer)
        self.t_weight = self.add_weight(
            shape=(self.units, self.units),
            name='t_weight',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.t_bias = self.add_weight(
            shape=(self.units,),
            name='t_bias',
            initializer="zeros",
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint)

    def _compute_carry_and_output(self, x, h_tm1, c_tm1, t_tm1):
        x_i, x_f, x_c, x_o = x
        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
        i = self.recurrent_activation(
            x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]) +
            self.input_gate_peephole_weights * t_tm1)
        f = self.recurrent_activation(x_f + K.dot(
            h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]) +
                                      self.forget_gate_peephole_weights * t_tm1)
        c = f * c_tm1 + i * self.activation(x_c + K.dot(
            h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]) +
                                            t_tm1*self.z_peephole_weights)
        o = self.recurrent_activation(
            x_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]) +
            self.output_gate_peephole_weights * c)
        return c, o

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        t_tm1 = inputs

        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=4)


        if 0 < self.dropout < 1.:
            inputs_i = inputs * dp_mask[0]
            inputs_f = inputs * dp_mask[1]
            inputs_c = inputs * dp_mask[2]
            inputs_o = inputs * dp_mask[3]
        else:
            inputs_i = inputs
            inputs_f = inputs
            inputs_c = inputs
            inputs_o = inputs
        k_i, k_f, k_c, k_o = array_ops.split(
            self.kernel, num_or_size_splits=4, axis=1)
        x_i = K.dot(inputs_i, k_i)
        x_f = K.dot(inputs_f, k_f)
        x_c = K.dot(inputs_c, k_c)
        x_o = K.dot(inputs_o, k_o)
        if self.use_bias:
            b_i, b_f, b_c, b_o = array_ops.split(
                self.bias, num_or_size_splits=4, axis=0)
            x_i = K.bias_add(x_i, b_i)
            x_f = K.bias_add(x_f, b_f)
            x_c = K.bias_add(x_c, b_c)
            x_o = K.bias_add(x_o, b_o)

        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1
        x = (x_i, x_f, x_c, x_o)
        h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
        c, o = self._compute_carry_and_output(x, h_tm1, c_tm1, t_tm1)

        h = o * self.activation(c)
        T = K.dot(h, self.t_weight)
        T = K.bias_add(T, self.t_bias)
        return T, [h, c]

class LSTMDecoder(tf.keras.layers.LSTM):
    def __init__(self,
                 units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if implementation == 0:
            logging.warning('`implementation=0` has been deprecated, '
                            'and now defaults to `implementation=1`.'
                            'Please update your layer call.')
        cell = LSTMCellDecoder(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            unit_forget_bias=unit_forget_bias,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            implementation=implementation)
        super(tf.keras.layers.LSTM, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = [InputSpec(ndim=3)]

    def bias_loss(self, target, output, axis=-1):

        target = tf.convert_to_tensor(target, dtype=self.dtype)  # (batch_size,length,label_size)

        temp1 = target[:, :, 2:] * 10
        temp2 = target[:, :, 0:2]
        target = tf.concat([temp2, temp1], axis=axis)

        return - tf.reduce_sum(target * tf.log(output), axis)


