# TF-IDF 说明文档
该章节代码负责人： 宋超 rogerspy@163.com

## 1、 功能

本算法是使用tf-idf对文本进行特征提取，然后使用朴素贝叶斯进行文本分类。这里我们使用20_new_group数据集，
下载地址在[这里](https://link.jianshu.com/?t=http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html),下载以后将解压后的文件放在data/目录下

## 2、调用

```
TfIdf()
```

### 2.1 调用实例

```
from tfidf import TfIdf

words_token, words_cont, docs_token = preprocess.counts(texts, docs)
co_maxtrix = tfidf.sparse_matrix(words_token, words_cont, docs_token, matrix_shape=(len(texts), words_token.shape[0]))
tfidf_matrix = tfidf.tfidf(co_maxtrix)
pred = tfidf.MultiNB(tfidf_matrix, label_array)
```

