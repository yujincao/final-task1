# 对比监督学习和自监督学习在图像分类任务上的性能表现

安装好python环境

#执行 pip install -r requirements.txt 安装依赖

训练可以再gpu或者cpu环境下

运行 pre_train.py
simclr自监督学习在自选的CIFAR-10数据集上训练ResNet-18
训练结果存储于runs下

运行 pre_train.py
CIFAR-100数据集中使用Linear Classification Protocol对其性能进行评测
评测结束保存loss和acc图像

运行 resnet18_train.py
CIFAR-100数据集在ResNet-18模型上从零开始以监督学习
训练后保存模型权重、loss和acc图像

运行 resnet18_imagenet_train.py
CIFAR-100数据集在ResNet-18模型上并且使用ImageNet数据集上采用监督学习训练得到的预训练权重训练
训练后保存模型权重、loss和acc图像



数据权重地址链接：https://pan.baidu.com/s/1XlnPHkNqy7Znr07sjxQLPw
提取码：pu7m
