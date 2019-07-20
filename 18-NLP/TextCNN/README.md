# TextCNN 算法说明

## 1、功能

本算法是用于文本分类的算法，算法来源于**Yoon Kim**2014年发表在EMNLP上的论文[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)。

## 2、调用

*textcnn.TextCNN(max_sent_len=50,
​                              max_word_num=50000,
​                              embedding_dims=100,
​                              class_num=10,
​                              last_activation='softmax',
​                              word_vector_matrix=None)*

### 2.1 参数说明

```
max_sent_len: 输入句子最大长度，
max_word_num: 词表大小，
embedding_dims: 词向量维度，
class_num: 文本类别数，
last_activation: 分类器激活函数，
word_vector_matrix: 预训练词向量矩阵
```

### 2.2 调用示例

```
from textcnn import TextCNN

model = TextCNN()
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])
model.fit(x_train, y_train, 
          batch_size=batch_size,
          epochs=epochs)
```

更加详细的方法可以查看`train.py`文件。



