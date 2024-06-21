import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.models as models
 
if torch.cuda.is_available():
    device=torch.device("cuda")
else:
    device=torch.device("cpu")
 
mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
 
transforms_fn=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std)
])
 
#100 classes containing 600 images each. There are 500 training images and 100 testing images per class.
#训练集
train_data=torchvision.datasets.CIFAR100('./data',train=True,transform=transforms_fn,download=True)
#测试集
test_data=torchvision.datasets.CIFAR100('./data',train=False,transform=transforms_fn,download=True)
 
train_data_size=len(train_data)
test_data_size=len(test_data)
print("训练数据集的长度为{}".format(train_data_size))
print("测试数据集的长度为{}".format(test_data_size))
 
# 试验输出
print(train_data.targets[0]) #输出第一个标签值，为19，对应牛的标签
print(type(train_data.targets)) # <class 'list'>,数据集标签类型是列表
print(train_data.data[0].shape) #(32, 32, 3) 原始数据集图像的维度
# plt.imshow(train_data.data[0]) #输出了牛的图片
# plt.show()
 
class_indices = list(range(10, 20)) + list(range(50, 60))+list(range(80,90)) #定义要抽取的类别序号
#根据类别序号抽取训练和测试样本
#CIFAR-100数据集的标签是从0开始的，所以类别序号[10, 19]对应于标签索引[10, 20)，
#类别序号[50, 59]对应于标签索引[50, 60)。确保你在定义类别序号时与数据集的类别对应。
#train_sampler和test_sampler，它们是自定义的采样器。采样器用于确定从数据集中选择哪些样本用于训练和测试。
#在这里，train_sampler和test_sampler分别使用train_indices和test_indices来选择相应的样本。
train_indices = [i for i in range(len(train_data)) if train_data.targets[i] in class_indices]
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices) #创建自定义的采样器，仅选择包含所选类别的样本
 
test_indices = [i for i in range(len(test_data)) if test_data.targets[i] in class_indices]
test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)
 
print(len(train_sampler))
print(len(test_sampler))
 
#利用dataloader来加载数据集
train= DataLoader(train_data, batch_size=1024, sampler=train_sampler) #在下面的源码解释中可以看出，如果sampler不为默认的None的时候，不用设置shuffle属性了
test = DataLoader(test_data, batch_size=1024, sampler=test_sampler)
 
 
 
examples = enumerate(test) #将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
batch_idx, (example_imgs, example_labels) = next(examples) #next的作用是返回迭代器的下一个项目,这里相当于取出来了test_data的第一个batch
print(batch_idx) #0
print(example_imgs[0].shape) #torch.Size([3, 32, 32]),经过DataLoader后，数据的维度会从32，32，3变为3，32，32，即神经网络能够接受的维度
print(example_labels[0].shape) #torch.Size([])
# fig = plt.figure()
# for i in range(64):
#     img=example_imgs[i] #3,32,32是原本图片数据的维度
#     img=np.transpose(img,(1,2,0)) #32,32,3需要转换成该维度才能输出3通道图片，即彩色图片
#     plt.subplot(8, 8, i + 1)
#     plt.imshow(img)
# # plt.savefig('CIFAR100')
# plt.show()
 
# examples = enumerate(test_data) #将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
# batch_idx, (example_data, example_targets) = next(examples)
# fig = plt.figure()
# for i in range(100):
#   plt.subplot(4,25,i+1)
#   plt.tight_layout()
#   plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#   plt.title("Ground Truth: {}".format(example_targets[i]))
#   plt.xticks([])
#   plt.yticks([])
# plt.show()

#构建RESNET18
class ResNet_18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet_18, self).__init__()
        
        # 使用预训练的 ResNet-18 模型
        self.backbone = models.resnet18(pretrained=True)
        
        # 修改第一个卷积层
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 修改最后一个全连接层
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
        # 添加 Dropout 层
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)
        x = self.dropout(x)

        return x
    
#网络模型
model=ResNet_18(num_classes=100)
model.to(device)
print(model)
#损失函数
loss_fn=nn.CrossEntropyLoss() #对于cross_entropy来说，他首先会对input进行log_softmax操作，然后再将log_softmax(input)的结果送入nll_loss；而nll_loss的input就是input。
#在多分类问题中，如果使用nn.CrossEntropyLoss()，则预测模型的输出层无需添加softmax层！！！
#如果是F.nll_loss，则需要添加softmax层!!!
loss_fn.to(device)
 
learning_rate=0.01
 
optimizer=torch.optim.SGD(params=model.parameters(),lr=learning_rate, momentum=0.9,weight_decay=0.0001)


train_acc_list = []
train_loss_list = []
test_acc_list = []
test_loss_list=[]
epochs=100
 
for epoch in range(epochs):
    print("-----第{}轮训练开始------".format(epoch + 1))
    train_loss=0.0
    test_loss=0.0
    train_sum,train_cor,test_sum,test_cor=0,0,0,0
 
    #训练步骤开始
    model.train()
    for batch_idx,(data,target) in enumerate(train):
        data,target=data.to(device),target.to(device)
 
        optimizer.zero_grad()  # 要将梯度清零，因为如果梯度不清零，pytorch中会将上次计算的梯度和本次计算的梯度累加
        # output = model(data)
        output = model(data)
        # loss = loss_fn(output, target)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()  # 更新所有的参数
 
        # 计算每轮训练集的Loss
        train_loss += loss.item()
 
        _, predicted = torch.max(output.data, 1)  # 选择最大的（概率）值所在的列数就是他所对应的类别数，
        # train_cor += (predicted == target).sum().item()  # 正确分类个数
        train_cor += (predicted == target).sum().item()  # 正确分类个数
        train_sum += target.size(0)  # train_sum+=predicted.shape[0]
 
    #测试步骤开始
    model.eval()
    # with torch.no_grad():
    for batch_idx1,(data,target) in enumerate(test):
        data, target = data.to(device), target.to(device)
 
        output = model(data)
        loss = loss_fn(output, target)
        test_loss+=loss.item()
        _, predicted = torch.max(output.data, 1)
        test_cor += (predicted == target).sum().item()
        test_sum += target.size(0)
 
    print("Train loss:{}   Train accuracy:{}%   Test loss:{}   Test accuracy:{}%".format(train_loss/batch_idx,100*train_cor/train_sum,
                                                                                       test_loss/batch_idx1,100*test_cor/test_sum))
    train_loss_list.append(train_loss / batch_idx)
    train_acc_list.append(100 * train_cor / train_sum)
    test_acc_list.append(100 * test_cor/ test_sum)
    test_loss_list.append(test_loss / batch_idx1)
 
#保存网络
torch.save(model,"ImagenetPre_CIFAR100_epoch{}.pth".format(epochs))
 
 
# Plot the accuracy and loss graphs
plt.figure(figsize=(16, 9))
plt.rcParams.update({'font.size': 14})

# Accuracy plot
plt.subplot(2, 1, 1)
plt.title('CIFAR-100 Accuracy Plot')
plt.plot(range(len(train_acc_list)), train_acc_list, c='b')
plt.plot(range(len(test_acc_list)), test_acc_list, c='g')
plt.legend(['Train Accuracy', 'Test Accuracy'])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)

# Loss plot
plt.subplot(2, 1, 2)
plt.title('CIFAR-100 Loss Plot')
plt.plot(range(len(train_loss_list)), train_loss_list, c='b')
plt.plot(range(len(test_loss_list)), test_loss_list, c='g')
plt.legend(['Train Loss', 'Test Loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

# Save the figure
plt.savefig('cifar100_imagenetPre_2_accuracy_loss.png', dpi=300)
# plt.show()