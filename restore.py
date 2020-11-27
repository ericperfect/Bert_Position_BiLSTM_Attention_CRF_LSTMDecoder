from kashgari.corpus import DataReader

from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config = tf.ConfigProto()

config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

test_x, test_y, _= DataReader.read_conll_format_file_word("/home/y182235017/law/predict.txt")
print(f"test data count: {len(test_x)}")

from kashgari import utils
model = utils.load_model("/home/y182235017/law/model/Word_BiLSTM_CRF_Model")
# model = load_model("/home/y182235017/law/model/Word_BiLSTM_CRF_Attention_Model_test1/my_model.h5")
import codecs
# result=model.evaluate(test_x,test_y,batch_size=128)
result = model.predict_entities_all(test_x)
with codecs.open("/home/y182235017/law/2.txt","w","utf-8") as file_obj:
    file_obj.write(result)