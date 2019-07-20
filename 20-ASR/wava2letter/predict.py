"""
Wav2Letter推理代码
Contributor：shaozhouce
Reviewer：xionglongfei
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
from Wav2Letter.model import Wav2Letter
from Wav2Letter.data import GoogleSpeechCommand
import keras.backend.tensorflow_backend as K

def eval(model, sample, sample_target):
        """
        计算一个单独样本的输出
        """
        _input = sample.reshape(1, sample.shape[0], sample.shape[1])
        log_prob = model.predict(_input)
        output = K.ctc_decode(log_prob, input_length=np.asarray(model.get_layer('pred').output_shape[1]).reshape(1,))
        with tf.Session() as sess:
            print("sample target", sample_target)
            print("predicted", output[0][0].eval())

if __name__ == "__main__":
    gs = GoogleSpeechCommand()
    inputs, targets, input_lengths= gs.load_vectors("./speech_data")
    index2char = gs.intencode.index2char

    print("random predict google speech dataset")
    print("data size", len(inputs))
    print("index2char", index2char)


    model = keras.models.load_model('./model/model.h5')
	
    samples_id = np.random.randint(0, len(inputs), size=5)    # 随机抽取5个样本进行预测
	
    for sample_id in samples_id:
        sample = inputs[sample_id]
        sample_target = targets[sample_id]
        eval(model, sample, sample_target)