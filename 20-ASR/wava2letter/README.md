# Wav2Letter with Tensorflow

Wav2Letter是由Facebooks AI Research (FAIR)发表的一个语音识别模型，具体的论文可以在这里看到 [paper](https://arxiv.org/pdf/1609.03193.pdf)。
<p align="center">
  <img src="Wav2Letter-diagram.png" alt="Precise 2 Diagram" height="700"/>
</p>
第二代Wav2Letter的论文 [paper](https://arxiv.org/abs/1712.09444)。 使用了Gated Convnets代替了普通的Convnets。



## 与论文的区别

* 使用了 CTC Loss
* 使用了 Greedy Decoder 

## Getting Started

## Requirements

```bash
pip install -r requirements.txt
```


## Data

使用 [Google Speech Command Dataset](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data). 进入网页后往下翻找Speech Commands Data Set v0.01数据集，这是一个包含许多1秒钟音频文件的轻量级数据集，每个1秒钟的音频文件包括1个英文单词。也可以直接使用(http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz)进行下载。

### download data

将文件下载并解压到 `./speech_data/speech_commands_v0.01/` 文件夹下。

### Prepare data


```bash
python Wav2Letter/data.py
```
`data.py` 文件会对解压好的数据进行预处理，生成模型训练所需要数据和标签，保存成 `./speech_data` 里的`x.npy` ， `y.npy` ，`x_length.npy`

## Train

```bash
python train.py --batch_size=256 --epochs=20
```
训练过程中会显示训练集的LOSS，每个epoch会显示训练集和测试集的loss，训练完后会使用测试集的第一个样本进行输出测试。

## Benchmark
训练需要大约5.5G的显存，在GTX970显卡上大约需要训练十几分钟。不同于图像相关任务中一个样本训练数据是大量的浮点数矩阵，经过MFCC提取的音频特征并不太占用存储空间，所以减小batch_size并不会明显降低显存的需求。训练时间？在CPU上训练模型比较慢。