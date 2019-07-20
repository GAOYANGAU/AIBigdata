# Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
此项目是基于Tensorflow、keras的[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial](https://arxiv.org/abs/1609.04802)实现。

## 网络架构
![SRGAN](srgan.png "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial")

## 损失函数 

### 损失分为内容损失与对抗损失
![loss](loss.png "loss")

### 内容损失
![loss](loss_content.png "content loss")

### 对抗损失
![loss](loss_adversarial.png "adversarial loss")

## 数据集
https://data.vision.ee.ethz.ch/cvl/DIV2K/

## 训练
```
python srgan.py
```

## 生成结果
### GPU: 1080TI
### batch_size = 8     
### epochs = 1000
### 耗时：5h

#### 低分辨率
![](images/9200_1_lowres.png)

#### 生成高辨率
![](images/9200_1_fake_hr.png)

#### 原始高分辨率
![](images/9200_1_imgs_hr.png)