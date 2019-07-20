# BiLSTM-CRF-NER模型文档说明


## 0、keras_contrib安装

该模型用到了keras_contrib(version=0.02)模块，请确保该模块已经安装，如未安装，安装方法如下：

```
pip install git+https://www.github.com/keras-team/keras-contrib.git
```

在此，也感谢`keras-contrib`团队的贡献，更多详细信息请访问[官方地址](https://www.github.com/keras-team/keras-contrib)。

## 1、 功能

该模型是利用双向LSTM模型进行命名实体识别。



## 2、调用

```
BiLSTM_CRF_NER(vocab_size=50000,
               emb_dims=100,
               birnn_unit=200,
               max_sent_len=200,
               tag_num=7)
```

### 2.1 参数说明

```
vocab_size: 词表大小
emb_dims: 词向量维度
birnn_unit: LSTM输出维度
max_sent_len: 最大文本长度
tag_num: 模型输出one-hot向量维度
```

### 2.2 调用实例

```
from bilstm_crf import BiLSTM_CRF_NER

bilstm_crf = BiLSTM_CRF_NER()
model = bilstm_crf.build_model()
model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_test, y_test))
```



