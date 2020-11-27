#!/usr/bin/env python
#-*-coding:utf-8-*-
##看一下标签
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

import kashgari

from kashgari.corpus import ChineseDailyNerCorpus
from kashgari.corpus import DataReader

# train_x, train_y = ChineseDailyNerCorpus.load_data('train')
# valid_x, valid_y = ChineseDailyNerCorpus.load_data('validate')
# test_x, test_y  = ChineseDailyNerCorpus.load_data('test')
train_x, train_y, train_z = DataReader.read_conll_format_file_word("/home/y182235017/law/trainwithseg.txt")
test_x, test_y, test_z = DataReader.read_conll_format_file_word("/home/y182235017/law/testwithseg.txt")

# print(f"train data count: {len(train_x)}")
# print(f"validate data count: {len(valid_x)}")
# print(f"test data count: {len(test_x)}")
from kashgari.embeddings import WordEmbedding
from kashgari.embeddings import BareEmbedding
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.labeling import BiLSTM_CRF_Model_Attention
from kashgari.tasks.labeling import CNN_BiLSTM_CRF_Model_WordSegmentation
from kashgari.tasks.labeling import BiLSTM_CRF_Model
from kashgari.tasks.labeling import BiLSTM_LSTMDecoder_Model
from kashgari.tasks.labeling import BiLSTM_CRF_Model_Position
from kashgari import callbacks_word


# bare_embed = BareEmbedding(task=kashgari.LABELING,sequence_length=500)
char_embed = WordEmbedding(w2v_path="/home/y182235017/law/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5",task=kashgari.LABELING,sequence_length=500)
# bert_embed = BERTEmbedding("/home/y182235017/law/chinese_L-12_H-768_A-12",task=kashgari.LABELING,sequence_length=500)
model = CNN_BiLSTM_CRF_Model_WordSegmentation(char_embed)
mycallback = callbacks_word.EvalCallBack(model,test_x,test_y,batch_size=128,path="/home/y182235017/law/model/Word_CNN_BiLSTM_CRF_Model_seg/")
mycallback={"callbacks":[mycallback]}
model.fit_without_generator_word(
          train_x,
          train_y,
          train_z,
          x_validate=test_x,
          y_validate=test_y,
          z_validate=test_z,
          epochs=20,
          batch_size=128,
          **mycallback)

# model.save("/root/BiLSTM_CRF_Model")

