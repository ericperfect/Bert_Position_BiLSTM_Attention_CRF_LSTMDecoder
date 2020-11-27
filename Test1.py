#!/usr/bin/env python
#-*-coding:utf-8-*-
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

train_x, train_y, = DataReader.read_conll_format_file("/home/y182235017/law/trainwithseg.txt")
test_x, test_y, = DataReader.read_conll_format_file("/home/y182235017/law/testwithseg.txt")

# print(f"train data count: {len(train_x)}")
from kashgari.embeddings import WordEmbedding
from kashgari.embeddings import BareEmbedding
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.labeling import BiLSTM_CRF_Model_Attention
from kashgari.tasks.labeling import Bert_Position_BiLSTM_Attention_CRF_LSTMDecoder_Model
from kashgari.tasks.labeling import BiLSTM_CRF_Model
from kashgari.tasks.labeling import CNN_BiLSTM_CRF_Model_Attenetion
from kashgari.tasks.labeling import CNN_BiLSTM_CRF_Model_Position
from kashgari.tasks.labeling import CNN_BiLSTM_CRF_Model
from kashgari.tasks.labeling import BiLSTM_LSTMDecoder_Model
from kashgari.tasks.labeling import Bert_BiLSTM_CRF_Model
from kashgari.tasks.labeling import BiLSTM_CRF_Model_Position
from kashgari import callbacks


# bare_embed = BareEmbedding(task=kashgari.LABELING,sequence_length=500)
# char_embed = WordEmbedding(w2v_path="/home/y182235017/law/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5",task=kashgari.LABELING,sequence_length=500)
bert_embed = BERTEmbedding("/home/y182235017/law/chinese_L-12_H-768_A-12",task=kashgari.LABELING,sequence_length=500)
model = Bert_Position_BiLSTM_Attention_CRF_LSTMDecoder_Model(bert_embed)
mycallback = callbacks.EvalCallBack(model,test_x,test_y,step=1,batch_size=128,path="/home/y182235017/law/model/Word_BiLSTM_CRF_Attention_Model_test1/")
mycallback={"callbacks":[mycallback]}
model.fit(train_x,
          train_y,
          x_validate=test_x,
          y_validate=test_y,
          epochs=40,
          batch_size=128,
          **mycallback)

# model.save("/home/y182235017/law/model/Word_BiLSTM_CRF_Attention_Model_test1/")
# from kashgari import utils
# model = utils.load_model("/home/y182235017/law/model/Word_BiLSTM_CRF_Attention_Model_test1/")
import codecs
result=model.evaluate(test_x,test_y,batch_size=128)
with codecs.open("/home/y182235017/law/2.txt","w","utf-8") as file_obj:
    file_obj.write(result)
