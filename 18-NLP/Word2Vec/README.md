# Word2Vec 说明文档
该章节代码负责人： 宋超 rogerspy@163.com

## 1、功能

本算法适用于训练词向量，算法来源于Mikolov等人2013年的论文[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)。需要注意的是，原始的Word2Vec算法包含CBOW和Skipgram以及在输出层有层级霍夫曼树和负采样两种，本模块只用keras实现了Skipgram+negative sampling。



## 2、调用

```
Word2Vec(vec_dims,
         vocab_size=None,
         window_size=3,
         negative_samples=5.,
         optimizer='rmsprop',
         loss='binary_crossentropy',
         batch_size=128,
         epochs=10)
```

### 2.1 参数说明

```
vec_dims: 词向量维度
vocab_size: 词表大小
window_size: 窗口大小
negative_samples: 负采样
optimizer: 优化器
loss: 损失函数
batch_size: 批大小
epochs: 训练轮数
```

### 2.2 调用示例

```
from word2vec import Word2Vec

word2vec = Word2Vec(vec_dims=100)
word_vec = word2vec.train(sentences, save_path='vectors.txt')
```

