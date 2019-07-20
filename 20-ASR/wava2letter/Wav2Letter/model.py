import os
import math
import keras
import tensorflow as tf
import numpy as np
from tensorflow import keras
from functools import reduce
from keras.layers import Conv1D
from keras.layers import ReLU
from keras.layers import Activation
from keras.layers import Input
from keras.models import Model
from keras.layers.core import Lambda
import keras.backend.tensorflow_backend as K

class Wav2Letter():

    def __init__(self, num_features, num_classes):
        super(Wav2Letter, self).__init__()
        self.inputs = Input(name='input', shape=(225, 13))
        
        

        self.x = Conv1D(filters=250, kernel_size=48, strides=2)(self.inputs)
        self.x = ReLU()(self.x)
        self.x = Conv1D(filters=250, kernel_size=7)(self.x)
        self.x = ReLU()(self.x)
        self.x = Conv1D(filters=250, kernel_size=7)(self.x)
        self.x = ReLU()(self.x)
        self.x = Conv1D(filters=250, kernel_size=7)(self.x)
        self.x = ReLU()(self.x)
        self.x = Conv1D(filters=250, kernel_size=7)(self.x)
        self.x = ReLU()(self.x)
        self.x = Conv1D(filters=250, kernel_size=7)(self.x)
        self.x = ReLU()(self.x)
        self.x = Conv1D(filters=250, kernel_size=7)(self.x)
        self.x = ReLU()(self.x)
        self.x = Conv1D(filters=250, kernel_size=7)(self.x)
        self.x = ReLU()(self.x)
        self.x = Conv1D(filters=2000, kernel_size=32)(self.x)
        self.x = ReLU()(self.x)
        self.x = Conv1D(filters=2000, kernel_size=1)(self.x)
        self.x = ReLU()(self.x)
        self.y_pred = Conv1D(name='pred', filters=num_classes+1, kernel_size=1)(self.x)

        self.log_probs = Activation('softmax', name='log_probs')(self.y_pred)

        self.targets = Input(name='target', shape=(6,))
        self.input_lengths = Input(name='input_length', shape=(1,))
        self.target_lengths = Input(name='label_length', shape=(1,))
        self.loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([self.log_probs, self.targets, self.input_lengths, self.target_lengths])
        
        self.model = Model(inputs=[self.inputs, self.targets, self.input_lengths, self.target_lengths], outputs=[self.loss_out])
  
        self.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')
        self.predict_model = Model(inputs=self.model.get_layer('input').input, outputs=self.model.get_layer('log_probs').output)
        self.model.summary()

    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args

        return K.ctc_batch_cost(y_true=labels, y_pred=y_pred, input_length=input_length, label_length=label_length)

    def fit(self, inputs, output, input_lengths, batch_size, epoch=15, print_every=50):
        """
        训练Wav2Letter模型，每个epoch训练结束的时候会在测试集上测试，输出loss。
        每个epoch训练结束会用测试集的第一个样本进行计算，观察模型输出
        """
        split_line = math.floor(0.9 * len(inputs))
        predict_lengths = np.ones((inputs.shape[0], 1)) * (self.model.get_layer('pred').output_shape[1])
        target_lengths = np.asarray([np.argwhere(output_ == -1)[0][0] if np.argwhere(output_ == -1).shape[0]>0 else output_.shape[0] for output_ in output]).reshape(-1,1)
        
        inputs_train, output_train = inputs[:split_line], output[:split_line]
        inputs_test, output_test = inputs[split_line:], output[split_line:]
        input_lengths_train, input_lengths_test = predict_lengths[:split_line], predict_lengths[split_line:]
        target_lengths_train, target_lengths_test = target_lengths[:split_line], target_lengths[split_line:]


        total_steps = math.ceil(len(inputs_train) / batch_size)

        for t in range(epoch):

            samples_processed = 0
            avg_epoch_loss = 0

            for step in range(total_steps):

                batch = \
                    inputs_train[samples_processed:batch_size + samples_processed]

                mini_batch_size = len(batch)
                targets = output_train[samples_processed: mini_batch_size + samples_processed]
                input_lengths = input_lengths_train[samples_processed: mini_batch_size + samples_processed]
                target_lengths = target_lengths_train[samples_processed: mini_batch_size + samples_processed]
   
                loss = self.model.train_on_batch([batch, targets, input_lengths, target_lengths], np.ones((mini_batch_size,1)))

                avg_epoch_loss += loss
                samples_processed += mini_batch_size

                if step % print_every == 0:
                    print("epoch", t + 1, ":" , "step", step + 1, "/", total_steps, ", loss ", loss.item())
                    
            #每个epoch训练结束后测试一次
            samples_processed = 0
            avg_epoch_test_loss = 0
            total_steps_test = math.ceil(len(inputs_test) / batch_size)
            for step in range(total_steps_test):
                batch = inputs_test[samples_processed:batch_size + samples_processed]
                mini_batch_size = len(batch)
                targets = output_test[samples_processed: mini_batch_size + samples_processed]

                input_lengths = input_lengths_test[samples_processed: mini_batch_size + samples_processed]
                target_lengths = target_lengths_test[samples_processed: mini_batch_size + samples_processed]
   
                test_loss = self.model.test_on_batch(
                    [batch, targets, input_lengths, target_lengths],
                    np.ones((mini_batch_size,1))
                    )
                avg_epoch_test_loss += test_loss
                samples_processed += mini_batch_size
            print("epoch", t + 1, "average epoch loss", avg_epoch_loss / total_steps)
            print("epoch", t + 1, "average epoch test_loss", avg_epoch_test_loss / total_steps_test)
        
        #训练完输出一个样本的测试结果
        sample = inputs_test[0]
        sample_target = output_test[0]
        self.eval(sample, sample_target)

    def eval(self, sample, sample_target):
        """
        计算一个单独样本的输出
        """
        
        _input = sample.reshape(1, sample.shape[0], sample.shape[1])
        log_prob = self.predict_model.predict(_input)
        output = K.ctc_decode(log_prob, input_length=np.asarray(self.model.get_layer('pred').output_shape[1]).reshape(1,))
        with tf.Session().as_default() as sess:
            print("sample target", sample_target)
            print("predicted", output[0][0].eval())
    
    def save(self, path):
        """
        保存训练好的模型
        """
        model_path = os.path.join(path, 'model.h5')
        if not os.path.exists(path):
            os.makedirs(path)
        self.predict_model.save(model_path)

