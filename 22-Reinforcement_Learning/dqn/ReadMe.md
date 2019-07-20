`打砖块游戏`

# 安装

```sh
pip3 install \
     --index-url http://mirrors.aliyun.com/pypi/simple \
     --trusted-host mirrors.aliyun.com  scikit-image==0.13.0 numpy==1.15.0 gym[atari]==0.12
```

# 训练

```sh
python3 train.py
```
注意: 如果需要在命令行执行的话, 使用 train1.py 进行训练. train1.py 脚本去掉了显示和日志部分

# 测试

```sh
python3 test.py
```
