import os
import numpy as np
import random
import pickle
from sonopy import mfcc_spec
from scipy.io import wavfile
from tqdm import tqdm


class IntegerEncode:
    """
    将labels编码成数字，不足max_label_seq长度的用-1补齐
    """

    def __init__(self, labels):

        self.char2index = {
            "pad":-1
        }
        self.index2char = {
            -1: "pad"
        }
        self.grapheme_count = 0 #字母统计
        self.process(labels)    #完成char2index和index2char
        self.max_label_seq = 6

    def process(self, labels):
        """
        用char2index和index2char两个字典纪录label到int和int到label的映射
        """
        strings = "".join(labels)
        for s in strings:
            if s not in self.char2index:
                self.char2index[s] = self.grapheme_count
                self.index2char[self.grapheme_count] = s
                self.grapheme_count += 1

    def convert_to_ints(self, label):
        """
        将label转成int
        """
        y = []
        for char in label:
            y.append(self.char2index[char])
        if len(y) < self.max_label_seq:
            diff = self.max_label_seq - len(y)
            pads = [self.char2index["pad"]] * diff
            y += pads
        return y

    def save(self, file_path):
        """
        将转换完成的label和相应的参数进行保存
        """
        file_name = os.path.join(file_path, "int_encoder.pkl")
        with open(file_name, 'wb') as f:
            pickle.dump(self.__dict__, f)


def normalize(values):
    """
    归一化到均值为0，标准差为1
    """
    return (values - np.mean(values)) / np.std(values)


class GoogleSpeechCommand():
    """
    用作数据集处理的一个类
    """

    def __init__(self, data_path="speech_data/speech_commands_v0.01", sample_rate=16000):
        self.data_path = data_path
        self.labels = [
            'right', 'eight', 'cat', 'tree', 'bed', 'happy', 'go', 'dog', 'no', 
            'wow', 'nine', 'left', 'stop', 'three', 'sheila', 'one', 'bird', 'zero',
            'seven', 'up', 'marvin', 'two', 'house', 'down', 'six', 'yes', 'on', 
            'five', 'off', 'four'
        ]
        self.intencode = IntegerEncode(self.labels)
        self.sr = sample_rate
        self.max_frame_len = 225

    def get_data(self, progress_bar=True):
        """
        返回音频的MFCC特征和相应的label
        """
        pg = tqdm if progress_bar else lambda x: x

        inputs, targets, input_lengths= [], [], []
        meta_data = []
        for label in self.labels:
            path = os.listdir(os.path.join(self.data_path, label))
            for audio in path:
                audio_path = os.path.join(self.data_path, label, audio)
                meta_data.append((audio_path, label))
        
        random.shuffle(meta_data)   #打乱数据集

        for md in pg(meta_data):
            audio_path = md[0]
            label = md[1]
            _, audio = wavfile.read(audio_path)
            mfccs = mfcc_spec(
                audio, self.sr, window_stride=(160, 80),
                fft_size=512, num_filt=20, num_coeffs=13
            )
            mfccs = normalize(mfccs)
            diff = self.max_frame_len - mfccs.shape[0]
            input_lengths.append(mfccs.shape[0])
            mfccs = np.pad(mfccs, ((0, diff), (0, 0)), "constant")#padding
            inputs.append(mfccs)

            target = self.intencode.convert_to_ints(label)
            targets.append(target)
        return inputs, targets, input_lengths

    @staticmethod
    def save_vectors(file_path, x, y, x_length):
        """
        将处理好的数据保存到x.npy，y.npy，x_length.npy
        """
        x_file = os.path.join(file_path, "x")
        y_file = os.path.join(file_path, "y")
        length_file = os.path.join(file_path, "x_length")
        np.save(x_file, np.asarray(x))
        np.save(y_file, np.asarray(y))
        np.save(length_file, np.asarray(x_length))

    @staticmethod
    def load_vectors(file_path):
        """
        加载训练数据和标签
        """
        x_file = os.path.join(file_path, "x.npy")
        y_file = os.path.join(file_path, "y.npy")
        length_file = os.path.join(file_path, "x_length.npy")
        inputs = np.load(x_file)
        targets = np.load(y_file)
        input_lengths = np.load(length_file)
        return inputs, targets, input_lengths


if __name__ == "__main__":
    gs = GoogleSpeechCommand()
    inputs, targets, input_lengths = gs.get_data()
    gs.save_vectors("./speech_data", inputs, targets, input_lengths)
    gs.intencode.save("./speech_data")
    print("preprocessed and saved")
