# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: models.py
# time: 2019-05-20 11:13

import logging
from typing import Dict, Any
from keras import regularizers
from tensorflow import keras
import tensorflow as tf
from kashgari.tasks.labeling.base_model import BaseLabelingModel
from kashgari.layers import L
from kashgari.layers.crf import CRF
from kashgari.layers.LSTMDecoder import LSTMDecoder
from kashgari.layers.attention_layer import Attention
from kashgari.layers.position_layer import Position_layer
from kashgari.layers.position_attention_layer1 import Position_attention_layer
from kashgari.utils import custom_objects
from kashgari.layers.bert_attention4 import bert_attention
custom_objects['CRF'] = CRF
custom_objects['LSTMDecoder'] = LSTMDecoder
custom_objects['Position_layer'] = Position_layer
custom_objects['Position_attention_layer'] = Position_attention_layer
custom_objects['bert_attention'] = bert_attention
# if tf.test.is_gpu_available(cuda_only=True):
#     L.LSTM = L.CuDNNLSTM

class BiLSTM_Model(BaseLabelingModel):
    """Bidirectional LSTM Sequence Labeling Model"""

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get hyper parameters of model
        Returns:
            hyper parameters dict
        """
        return {
            'layer_blstm': {
                'units': 128,
                'return_sequences': True
            },
            'layer_dropout': {
                'rate': 0.4
            },
            'layer_time_distributed': {},
            'layer_activation': {
                'activation': 'softmax'
            }
        }

    def build_model_arc(self):
        """
        build model architectural
        """
        output_dim = len(self.pre_processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layer_blstm = L.Bidirectional(L.LSTM(**config['layer_blstm']),#units=128 return_sequence = Ture
                                      name='layer_blstm')

        layer_dropout = L.Dropout(**config['layer_dropout'],
                                  name='layer_dropout')

        layer_time_distributed = L.TimeDistributed(L.Dense(output_dim,
                                                           **config['layer_time_distributed']),
                                                   name='layer_time_distributed')
        layer_activation = L.Activation(**config['layer_activation'])

        tensor = layer_blstm(embed_model.output)
        tensor = layer_dropout(tensor)
        tensor = layer_time_distributed(tensor)
        output_tensor = layer_activation(tensor)

        self.tf_model = keras.Model(embed_model.inputs, output_tensor)


class BiLSTM_CRF_Model(BaseLabelingModel):
    """Bidirectional LSTM CRF Sequence Labeling Model"""

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get hyper parameters of model
        Returns:
            hyper parameters dict
        """
        return {
            'layer_blstm': {
                'units': 128,
                'return_sequences': True
            },
            'layer_dense': {
                'units': 64,
                'activation': 'tanh'
            }
        }

    def build_model_arc(self):
        """
        build model architectural
        """
        output_dim = len(self.pre_processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        # layer_blstm = L.Bidirectional(L.LSTM(**config['layer_blstm']),
        #                               name='layer_blstm')
        layer_blstm = L.Bidirectional(L.CuDNNLSTM(**config['layer_blstm']),
                                      name='layer_blstm')

        layer_dense = L.Dense(**config['layer_dense'], name='layer_dense')
        layer_crf_dense = L.Dense(output_dim, name='layer_crf_dense')
        layer_crf = CRF(output_dim, name='layer_crf') #全局定制类

        tensor = layer_blstm(embed_model.output)
        tensor = layer_dense(tensor)
        tensor = layer_crf_dense(tensor)
        output_tensor = layer_crf(tensor)

        self.layer_crf = layer_crf
        self.tf_model = keras.Model(embed_model.inputs, output_tensor)

    def compile_model(self, **kwargs):
        if kwargs.get('loss') is None:
            kwargs['loss'] = self.layer_crf.loss
        if kwargs.get('metrics') is None:
            kwargs['metrics'] = [self.layer_crf.viterbi_accuracy]
        super(BiLSTM_CRF_Model, self).compile_model(**kwargs)

class Bert_BiLSTM_CRF_Model(BaseLabelingModel):
    """Bidirectional LSTM CRF Sequence Labeling Model"""

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get hyper parameters of model
        Returns:
            hyper parameters dict
        """
        return {
            'layer_blstm': {
                'units': 128,
                'return_sequences': True
            },
            'layer_dense': {
                'units': 64,
                'activation': 'tanh'
            }
        }

    def build_model_arc(self):
        """
        build model architectural
        """
        output_dim = len(self.pre_processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model


        layer_bert = bert_attention(name='layer_bert')
        layer_blstm = L.Bidirectional(L.CuDNNLSTM(**config['layer_blstm']),
                                      name='layer_blstm')
        layer_dense = L.Dense(**config['layer_dense'], name='layer_dense')
        layer_crf_dense = L.Dense(output_dim, name='layer_crf_dense')
        layer_crf = CRF(output_dim, name='layer_crf') #全局定制类

        tensor = layer_bert(embed_model.output)
        tensor = layer_blstm(tensor)
        tensor = layer_dense(tensor)
        tensor = layer_crf_dense(tensor)
        output_tensor = layer_crf(tensor)

        self.layer_crf = layer_crf
        self.tf_model = keras.Model(embed_model.inputs, output_tensor)

    def compile_model(self, **kwargs):
        if kwargs.get('loss') is None:
            kwargs['loss'] = self.layer_crf.loss
        if kwargs.get('metrics') is None:
            kwargs['metrics'] = [self.layer_crf.viterbi_accuracy]
        super(Bert_BiLSTM_CRF_Model, self).compile_model(**kwargs)

class BiLSTM_CRF_Model_Attention(BiLSTM_CRF_Model):
    """Bidirectional LSTM CRF Sequence Labeling Model"""

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get hyper parameters of model
        Returns:
            hyper parameters dict
        """
        return {
            'layer_blstm': {
                'units': 128,
                'return_sequences': True
            },
            'layer_dense1': {
                'units': 128,
                'activation': 'tanh'
            },
            'layer_dense2': {
                'units': 64,
                'activation': 'tanh'
            }
        }

    def build_model_arc(self):
        """
        build model architectural
        """
        output_dim = len(self.pre_processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model


        layer_blstm = L.Bidirectional(L.CuDNNLSTM(**config['layer_blstm']),
                                      name='layer_blstm')
        layer_attention = Attention(name='layer_attention')
        layer_Activation = L.Activation("tanh", name="layer_Activation")
        layer_dense1 = L.Dense(**config['layer_dense1'], name='layer_dense1')
        layer_dense2 = L.Dense(**config['layer_dense2'], name='layer_dense2')
        layer_crf_dense = L.Dense(output_dim, name='layer_crf_dense')
        layer_crf = CRF(output_dim, name='layer_crf') #全局定制类

        tensor = layer_blstm(embed_model.output)
        tensor = layer_attention(tensor)
        tensor = layer_Activation(tensor)
        tensor = layer_dense1(tensor)
        tensor = layer_dense2(tensor)
        tensor = layer_crf_dense(tensor)
        output_tensor = layer_crf(tensor)

        self.layer_crf = layer_crf
        self.tf_model = keras.Model(embed_model.inputs, output_tensor)

class BiLSTM_CRF_Model_Position(BaseLabelingModel):
    """Bidirectional LSTM CRF Sequence Labeling Model"""

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get hyper parameters of model
        Returns:
            hyper parameters dict
        """
        return {
            'layer_blstm': {
                'units': 128,
                'return_sequences': True
            },
            'layer_dense': {
                'units': 64,
                'activation': 'tanh'
            }
        }

    def build_model_arc(self):
        """
        build model architectural
        """
        output_dim = len(self.pre_processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layer_position = Position_attention_layer(name='layer_position')
        layer_blstm = L.Bidirectional(L.CuDNNLSTM(**config['layer_blstm']),
                                       name='layer_blstm')

        layer_dense = L.Dense(**config['layer_dense'], name='layer_dense')
        layer_crf_dense = L.Dense(output_dim, name='layer_crf_dense')
        layer_crf = CRF(output_dim, name='layer_crf') #全局定制类

        tensor = layer_position(embed_model.output)
        tensor = layer_blstm(tensor)
        tensor = layer_dense(tensor)
        tensor = layer_crf_dense(tensor)
        output_tensor = layer_crf(tensor)

        self.layer_crf = layer_crf
        self.tf_model = keras.Model(embed_model.inputs, output_tensor)

    def compile_model(self, **kwargs):
        if kwargs.get('loss') is None:
            kwargs['loss'] = self.layer_crf.loss
        if kwargs.get('metrics') is None:
            kwargs['metrics'] = [self.layer_crf.viterbi_accuracy]
        super(BiLSTM_CRF_Model_Position, self).compile_model(**kwargs)

class Bert_Position_BiLSTM_Attention_CRF_LSTMDecoder_Model(BaseLabelingModel):
    """Bidirectional LSTM CRF Sequence Labeling Model"""

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get hyper parameters of model
        Returns:
            hyper parameters dict
        """
        return {
            'layer_blstm': {
                'units': 128,
                'return_sequences': True
            },
            'layer_dense1': {
                'units': 128,
                'activation': 'tanh'
            },
            'layer_dense2': {
                'units': 64,
                'activation': 'tanh'
            },
            'layer_LSTMDecoder': {
                'units': 256,
                'return_sequences': True
            }
        }

    def build_model_arc(self):
        """
        build model architectural
        """
        output_dim = len(self.pre_processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model


        layer_bert = bert_attention(name='layer_bert')
        layer_position = Position_attention_layer(name='layer_position')
        layer_blstm = L.Bidirectional(L.CuDNNLSTM(**config['layer_blstm']),
                                      name='layer_blstm')
        layer_LSTMDecoder = LSTMDecoder(**config['layer_LSTMDecoder'], name='layer_LSTMDecoder')
        layer_attention = Attention(name='layer_attention')
        layer_Activation = L.Activation("tanh", name="layer_Activation")
        layer_dense1 = L.Dense(**config['layer_dense1'], name='layer_dense1')
        layer_dense2 = L.Dense(**config['layer_dense2'], name='layer_dense2')

        layer_crf_dense = L.Dense(output_dim, name='layer_crf_dense')
        layer_crf = CRF(output_dim, name='layer_crf') #全局定制类

        tensor = layer_bert(embed_model.output)
        tensor = layer_position(tensor)
        tensor = layer_blstm(tensor)
        tensor = layer_LSTMDecoder(tensor)
        tensor = layer_attention(tensor)
        tensor = layer_Activation(tensor)
        tensor = layer_dense1(tensor)
        tensor = layer_dense2(tensor)
        tensor = layer_crf_dense(tensor)
        output_tensor = layer_crf(tensor)

        self.layer_crf = layer_crf
        self.tf_model = keras.Model(embed_model.inputs, output_tensor)

    def compile_model(self, **kwargs):
        if kwargs.get('loss') is None:
            kwargs['loss'] = self.layer_crf.loss
        if kwargs.get('metrics') is None:
            kwargs['metrics'] = [self.layer_crf.viterbi_accuracy]
        super(Bert_Position_BiLSTM_Attention_CRF_LSTMDecoder_Model, self).compile_model(**kwargs)

class CNN_BiLSTM_CRF_Model(BaseLabelingModel):
    """Bidirectional LSTM CRF Sequence Labeling Model"""

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get hyper parameters of model
        Returns:
            hyper parameters dict
        """
        return {
            'layer_conv2': {
                'filters': 32,
                'kernel_size': 2,
                'padding': 'same',
                'activation': 'relu',

            },
            'layer_conv3': {
                'filters': 32,
                'kernel_size': 3,
                'padding': 'same',
                'activation': 'relu',
            },
            'layer_conv4': {
                'filters': 32,
                'kernel_size': 4,
                'padding': 'same',
                'activation': 'relu',
            },
            'layer_conv5': {
                'filters': 32,
                'kernel_size': 5,
                'padding': 'same',
                'activation': 'relu',
            },
            'layer_conv6': {
                'filters': 32,
                'kernel_size': 6,
                'padding': 'same',
                'activation': 'relu',
            },
            'layer_conv7': {
                'filters': 32,
                'kernel_size': 7,
                'padding': 'same',
                'activation': 'relu',
            },
            'layer_conv8': {
                'filters': 32,
                'kernel_size': 8,
                'padding': 'same',
                'activation': 'relu',
            },
            'layer_conv9': {
                'filters': 32,
                'kernel_size': 9,
                'padding': 'same',
                'activation': 'relu',
            },
            'layer_conv10': {
                'filters': 32,
                'kernel_size': 10,
                'padding': 'same',
                'activation': 'relu',
            },
            'layer_blstm': {
                'units': 128,
                'return_sequences': True
            },
            'layer_dense': {
                'units': 64,
                'activation': 'tanh'
            }
        }

    def build_model_arc(self):
        """
        build model architectural
        """
        output_dim = len(self.pre_processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        # layer_blstm = L.Bidirectional(L.LSTM(**config['layer_blstm']),
        #                               name='layer_blstm')
        layer_conv2 = L.Conv1D(**config['layer_conv2'],
                              name='layer_conv2',
                               kernel_regularizer=regularizers.l2(0.01))
        layer_conv3 = L.Conv1D(**config['layer_conv3'],
                               name='layer_conv3',
                               kernel_regularizer=regularizers.l2(0.02))
        layer_conv4 = L.Conv1D(**config['layer_conv4'],
                               name='layer_conv4',
                               kernel_regularizer=regularizers.l2(0.03))
        layer_conv5 = L.Conv1D(**config['layer_conv5'],
                               name='layer_conv5',
                               kernel_regularizer=regularizers.l2(0.04))
        layer_conv6 = L.Conv1D(**config['layer_conv6'],
                               name='layer_conv6',
                               kernel_regularizer=regularizers.l2(0.05))
        layer_conv7 = L.Conv1D(**config['layer_conv7'],
                               name='layer_conv7',
                               kernel_regularizer=regularizers.l2(0.06))
        layer_conv8 = L.Conv1D(**config['layer_conv8'],
                               name='layer_conv8',
                               kernel_regularizer=regularizers.l2(0.07))
        layer_conv9 = L.Conv1D(**config['layer_conv9'],
                               name='layer_conv9',
                               kernel_regularizer=regularizers.l2(0.08))
        layer_conv10 = L.Conv1D(**config['layer_conv10'],
                               name='layer_conv10',
                               kernel_regularizer=regularizers.l2(0.09))

        layer_blstm = L.Bidirectional(L.CuDNNLSTM(**config['layer_blstm']),
                                      name='layer_blstm')
        layer_dense = L.Dense(**config['layer_dense'], name='layer_dense')
        layer_crf_dense = L.Dense(output_dim, name='layer_crf_dense')
        layer_crf = CRF(output_dim, name='layer_crf') #全局定制类
        tensor2 = layer_conv2(embed_model.output)
        tensor3 = layer_conv3(embed_model.output)
        tensor4 = layer_conv4(embed_model.output)
        tensor5 = layer_conv5(embed_model.output)
        tensor6 = layer_conv6(embed_model.output)
        tensor7 = layer_conv7(embed_model.output)
        tensor8 = layer_conv8(embed_model.output)
        tensor9 = layer_conv9(embed_model.output)
        tensor10 = layer_conv10(embed_model.output)
        tensor = keras.layers.concatenate([tensor2, tensor3, tensor4, tensor5,tensor6, tensor7,tensor8, tensor9,tensor10], 2)
        tensor = layer_blstm(tensor)
        tensor = layer_dense(tensor)
        tensor = layer_crf_dense(tensor)
        output_tensor = layer_crf(tensor)

        self.layer_crf = layer_crf
        self.tf_model = keras.Model(embed_model.inputs, output_tensor)

    def compile_model(self, **kwargs):
        if kwargs.get('loss') is None:
            kwargs['loss'] = self.layer_crf.loss
        if kwargs.get('metrics') is None:
            kwargs['metrics'] = [self.layer_crf.viterbi_accuracy]
        super(CNN_BiLSTM_CRF_Model, self).compile_model(**kwargs)

class CNN_BiLSTM_CRF_Model_Position(BaseLabelingModel):
    """Bidirectional LSTM CRF Sequence Labeling Model"""

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get hyper parameters of model
        Returns:
            hyper parameters dict
        """
        return {
            'layer_conv': {
                'filters': 32,
                'kernel_size': 3,
                'padding': 'same',
                'activation': 'relu'
            },
            'layer_blstm': {
                'units': 128,
                'return_sequences': True
            },
            'layer_dense': {
                'units': 64,
                'activation': 'tanh'
            }
        }

    def build_model_arc(self):
        """
        build model architectural
        """
        output_dim = len(self.pre_processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        # layer_blstm = L.Bidirectional(L.LSTM(**config['layer_blstm']),
        #                               name='layer_blstm')
        layer_conv = L.Conv1D(**config['layer_conv'],
                              name='layer_conv')
        layer_position = Position_layer(name='layer_position')
        layer_blstm = L.Bidirectional(L.CuDNNLSTM(**config['layer_blstm']),
                                      name='layer_blstm')
        layer_dense = L.Dense(**config['layer_dense'], name='layer_dense')
        layer_crf_dense = L.Dense(output_dim, name='layer_crf_dense')
        layer_crf = CRF(output_dim, name='layer_crf') #全局定制类
        tensor = layer_conv(embed_model.output)
        tensor = layer_position(tensor)
        tensor = layer_blstm(tensor)
        tensor = layer_dense(tensor)
        tensor = layer_crf_dense(tensor)
        output_tensor = layer_crf(tensor)

        self.layer_crf = layer_crf
        self.tf_model = keras.Model(embed_model.inputs, output_tensor)

    def compile_model(self, **kwargs):
        if kwargs.get('loss') is None:
            kwargs['loss'] = self.layer_crf.loss
        if kwargs.get('metrics') is None:
            kwargs['metrics'] = [self.layer_crf.viterbi_accuracy]
        super(CNN_BiLSTM_CRF_Model_Position, self).compile_model(**kwargs)

class CNN_BiLSTM_CRF_Model_Attenetion(BaseLabelingModel):
    """Bidirectional LSTM CRF Sequence Labeling Model"""

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get hyper parameters of model
        Returns:
            hyper parameters dict
        """
        return {
            'layer_conv': {
                'filters': 32,
                'kernel_size': 3,
                'padding': 'same',
                'activation': 'relu'
            },
            'layer_blstm': {
                'units': 128,
                'return_sequences': True
            },
            'layer_dense': {
                'units': 64,
                'activation': 'tanh'
            }
        }

    def build_model_arc(self):
        """
        build model architectural
        """
        output_dim = len(self.pre_processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        # layer_blstm = L.Bidirectional(L.LSTM(**config['layer_blstm']),
        #                               name='layer_blstm')
        layer_conv = L.Conv1D(**config['layer_conv'],
                              name='layer_conv')
        layer_blstm = L.Bidirectional(L.CuDNNLSTM(**config['layer_blstm']),
                                      name='layer_blstm')
        layer_dense = L.Dense(**config['layer_dense'], name='layer_dense')
        layer_attention = Attention(name='layer_attention')
        layer_Activation = L.Activation("tanh", name="layer_Activation")

        layer_crf_dense = L.Dense(output_dim, name='layer_crf_dense')
        layer_crf = CRF(output_dim, name='layer_crf') #全局定制类

        tensor = layer_conv(embed_model.output)
        tensor = layer_blstm(tensor)
        tensor = layer_dense(tensor)
        tensor = layer_attention(tensor)
        tensor = layer_Activation(tensor)
        tensor = layer_crf_dense(tensor)
        output_tensor = layer_crf(tensor)

        self.layer_crf = layer_crf
        self.tf_model = keras.Model(embed_model.inputs, output_tensor)

    def compile_model(self, **kwargs):
        if kwargs.get('loss') is None:
            kwargs['loss'] = self.layer_crf.loss
        if kwargs.get('metrics') is None:
            kwargs['metrics'] = [self.layer_crf.viterbi_accuracy]
        super(CNN_BiLSTM_CRF_Model_Attenetion, self).compile_model(**kwargs)

class BiLSTM_LSTMDecoder_Model(BaseLabelingModel):
    """Bidirectional LSTM CRF Sequence Labeling Model"""

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get hyper parameters of model
        Returns:
            hyper parameters dict
        """
        return {
            'layer_blstm': {
                'units': 128,
                'return_sequences': True
            },

            'layer_LSTMDecoder': {
                'units': 256,
                'return_sequences': True
            },
            'layer_dense': {
                'units': 64,
                'activation': 'tanh'
            }
        }


    def build_model_arc(self):
        """
        build model architectural
        """
        output_dim = len(self.pre_processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layer_blstm = L.Bidirectional(L.LSTM(**config['layer_blstm']),
                                      name='layer_blstm')

        layer_LSTMDecoder = LSTMDecoder(**config['layer_LSTMDecoder'], name='layer_LSTMDecoder')
        layer_dense = L.Dense(**config['layer_dense'], name='layer_dense')
        layer_decoder_dense = L.Dense(output_dim, name='layer_decoder_dense')
        softmax_layer = L.Activation(tf.nn.softmax, name="softmax_layer")

        tensor = layer_blstm(embed_model.output)
        tensor = layer_LSTMDecoder(tensor)
        tensor = layer_dense(tensor)
        tensor = layer_decoder_dense(tensor)
        output_tensor = softmax_layer(tensor)

        self.layer_LSTMDecoder = layer_LSTMDecoder
        self.tf_model = keras.Model(embed_model.inputs, output_tensor)

    def compile_model(self, **kwargs):
        if kwargs.get('loss') is None:
            kwargs['loss'] = self.layer_LSTMDecoder.bias_loss
        super(BiLSTM_LSTMDecoder_Model, self).compile_model(**kwargs)

class BiLSTM_LSTMDecoder_CRF_Model(BaseLabelingModel):
    """Bidirectional LSTM CRF Sequence Labeling Model"""

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get hyper parameters of model
        Returns:
            hyper parameters dict
        """
        return {
            'layer_blstm': {
                'units': 128,
                'return_sequences': True
            },

            'layer_LSTMDecoder': {
                'units': 256,
                'return_sequences': True
            },
            'layer_dense': {
                'units': 64,
                'activation': 'tanh'
            }
        }


    def build_model_arc(self):
        """
        build model architectural
        """
        output_dim = len(self.pre_processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layer_blstm = L.Bidirectional(L.LSTM(**config['layer_blstm']),
                                      name='layer_blstm')

        layer_LSTMDecoder = LSTMDecoder(**config['layer_LSTMDecoder'], name='layer_LSTMDecoder')
        layer_dense = L.Dense(**config['layer_dense'], name='layer_dense')
        layer_decoder_dense = L.Dense(output_dim, name='layer_decoder_dense')
        layer_crf = CRF(output_dim, name='layer_crf')  # 全局定制类


        tensor = layer_blstm(embed_model.output)
        tensor = layer_LSTMDecoder(tensor)
        tensor = layer_dense(tensor)
        tensor = layer_decoder_dense(tensor)
        output_tensor = layer_crf(tensor)

        self.layer_crf = layer_crf
        self.tf_model = keras.Model(embed_model.inputs, output_tensor)

    def compile_model(self, **kwargs):
        if kwargs.get('loss') is None:
            kwargs['loss'] = self.layer_crf.loss
        super(BiLSTM_LSTMDecoder_CRF_Model, self).compile_model(**kwargs)



class CNN_BiLSTM_CRF_Model_WordSegmentation(BiLSTM_CRF_Model):
    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get hyper parameters of model
        Returns:
            hyper parameters dict
        """
        return {
            'layer_conv': {
                'filters': 32,
                'kernel_size': 3,
                'padding': 'same',
                'activation': 'relu'
            },

            'layer_blstm': {
                'units': 128,
                'return_sequences': True
            },
            'layer_dense': {
                'units': 64,
                'activation': 'tanh'
            }
        }

    def build_model_arc(self):
        """
        build model architectural
        """
        output_dim = len(self.pre_processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layer_conv = L.Conv1D(**config['layer_conv'],
                              name='layer_conv')
        layer_blstm = L.Bidirectional(L.CuDNNLSTM(**config['layer_blstm']),
                                      name='layer_blstm')

        layer_dense = L.Dense(**config['layer_dense'], name='layer_dense')
        layer_crf_dense1 = L.Dense(output_dim, name='layer_crf_dense')
        layer_crf1 = CRF(output_dim, name='layer_crf1') #全局定制类

        layer_crf_dense2 = L.Dense(output_dim, name='layer_crf_dense2')
        layer_crf2 = CRF(output_dim, name='layer_crf2')

        tensor = layer_conv(embed_model.output)
        tensor1 = layer_crf_dense1(tensor)
        output_tensor1 = layer_crf1(tensor1)
        tensor = layer_blstm(tensor)
        tensor = layer_dense(tensor)
        tensor2 = layer_crf_dense2(tensor)
        output_tensor2 = layer_crf2(tensor2)

        self.layer_crf1 = layer_crf1
        self.layer_crf2 = layer_crf2
        self.tf_model = keras.Model(inputs = embed_model.inputs, outputs = [output_tensor2,output_tensor1])

    def compile_model(self, **kwargs):
        if kwargs.get('loss') is None:
            kwargs['loss'] = [self.layer_crf1.loss,self.layer_crf2.loss]
        if kwargs.get('metrics') is None:
            kwargs['metrics'] = [self.layer_crf1.viterbi_accuracy,self.layer_crf2.viterbi_accuracy]
        if kwargs.get('loss_weights') is None:
            kwargs['loss_weights'] = [0.6,0.4]
        super(BiLSTM_CRF_Model, self).compile_model(**kwargs)


class BiGRU_Model(BaseLabelingModel):
    """Bidirectional GRU Sequence Labeling Model"""

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get hyper parameters of model
        Returns:
            hyper parameters dict
        """
        return {
            'layer_bgru': {
                'units': 128,
                'return_sequences': True
            },
            'layer_dropout': {
                'rate': 0.4
            },
            'layer_time_distributed': {},
            'layer_activation': {
                'activation': 'softmax'
            }
        }

    def build_model_arc(self):
        """
        build model architectural
        """
        output_dim = len(self.pre_processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layer_blstm = L.Bidirectional(L.GRU(**config['layer_bgru']),
                                      name='layer_bgru')

        layer_dropout = L.Dropout(**config['layer_dropout'],
                                  name='layer_dropout')

        layer_time_distributed = L.TimeDistributed(L.Dense(output_dim,
                                                           **config['layer_time_distributed']),
                                                   name='layer_time_distributed')
        layer_activation = L.Activation(**config['layer_activation'])

        tensor = layer_blstm(embed_model.output)
        tensor = layer_dropout(tensor)
        tensor = layer_time_distributed(tensor)
        output_tensor = layer_activation(tensor)

        self.tf_model = keras.Model(embed_model.inputs, output_tensor)


class BiGRU_CRF_Model(BaseLabelingModel):
    """Bidirectional GRU CRF Sequence Labeling Model"""

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get hyper parameters of model
        Returns:
            hyper parameters dict
        """
        return {
            'layer_bgru': {
                'units': 128,
                'return_sequences': True
            },
            'layer_dense': {
                'units': 64,
                'activation': 'tanh'
            }
        }

    def build_model_arc(self):
        """
        build model architectural
        """
        output_dim = len(self.pre_processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layer_blstm = L.Bidirectional(L.GRU(**config['layer_bgru']),
                                      name='layer_bgru')

        layer_dense = L.Dense(**config['layer_dense'], name='layer_dense')
        layer_crf_dense = L.Dense(output_dim, name='layer_crf_dense')
        layer_crf = CRF(output_dim, name='layer_crf')

        tensor = layer_blstm(embed_model.output)
        tensor = layer_dense(tensor)
        tensor = layer_crf_dense(tensor)
        output_tensor = layer_crf(tensor)

        self.layer_crf = layer_crf
        self.tf_model = keras.Model(embed_model.inputs, output_tensor)

    def compile_model(self, **kwargs):
        if kwargs.get('loss') is None:
            kwargs['loss'] = self.layer_crf.loss
        if kwargs.get('metrics') is None:
            kwargs['metrics'] = [self.layer_crf.viterbi_accuracy]
        super(BiGRU_CRF_Model, self).compile_model(**kwargs)


class CNN_LSTM_Model(BaseLabelingModel):
    """CNN LSTM Sequence Labeling Model"""

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get hyper parameters of model
        Returns:
            hyper parameters dict
        """
        return {
            'layer_conv': {
                'filters': 32,
                'kernel_size': 3,
                'padding': 'same',
                'activation': 'relu'
            },
            'layer_lstm': {
                'units': 128,
                'return_sequences': True
            },
            'layer_dropout': {
                'rate': 0.4
            },
            'layer_time_distributed': {},
            'layer_activation': {
                'activation': 'softmax'
            }
        }

    def build_model_arc(self):
        """
        build model architectural
        """
        output_dim = len(self.pre_processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layer_conv = L.Conv1D(**config['layer_conv'],
                              name='layer_conv')
        layer_lstm = L.LSTM(**config['layer_lstm'],
                            name='layer_lstm')
        layer_dropout = L.Dropout(**config['layer_dropout'],
                                  name='layer_dropout')
        layer_time_distributed = L.TimeDistributed(L.Dense(output_dim,
                                                           **config['layer_time_distributed']),
                                                   name='layer_time_distributed')
        layer_activation = L.Activation(**config['layer_activation'])

        tensor = layer_conv(embed_model.output)
        tensor = layer_lstm(tensor)
        tensor = layer_dropout(tensor)
        tensor = layer_time_distributed(tensor)
        output_tensor = layer_activation(tensor)

        self.tf_model = keras.Model(embed_model.inputs, output_tensor)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    from kashgari.corpus import ChineseDailyNerCorpus

    valid_x, valid_y = ChineseDailyNerCorpus.load_data('train')

    model = BiLSTM_CRF_Model()
    model.fit(valid_x, valid_y, epochs=50, batch_size=64)
    model.evaluate(valid_x, valid_y)
