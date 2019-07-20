## Requirements##
- tensorflow 1.10.0
- python 3.6 
- numpy 
- scipy 

#文件结构：
checkpoints/用来存放训练好的权重
examples/ 用来存放训练和测试所需数据
--content/ 用来测试的数据
--style/ 用来存放风格照片style image path
data/coco/train2014  用来存放训练集  coco:http://msvocds.blob.core.windows.net/coco2014/train2014.zip
premodel/用来存放VGG19的权重
下载地址：http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat

#快速上手：测试自己的图片
python evaluate.py --checkpoint_dir path/to/checkpoints/folder   --in_path path/to/test/image/folder
例如 python evaluate.py --checkpoint_dir checkpoints/la_muse   --in_path examples/content
其中la_muse是训练出来的一种风格模型的checkpoint目录
推理的结果被保存在examples/result里
这些参数也可以在evaluate.py 里修改，然后直接运行

#训练自己的风格图片
##在main.py中，可以设置好自己的超参数
##要想更加详细的了解训练过程查看solver.py
python main.py --style_img 自己的风格图片.jpg \
  --train_path 训练集路径，一般是用coco train2014数据库 \
  --test_path path/to/test/data/fold \
  --vgg_path path/to/vgg19/imagenet-vgg-verydeep-19.mat
