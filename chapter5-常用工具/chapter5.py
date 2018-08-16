import torch as t
from torch.utils import data
import os
from PIL import  Image
import numpy as np
from torchvision.datasets import ImageFolder


class DogCat(data.Dataset):
    def __init__(self, root):
        imgs = os.listdir(root)
        # print(imgs)
        # 所有图片的绝对路径
        # 这里不实际加载图片，只是指定路径，当调用__getitem__时才会真正读图片
        self.imgs = [os.path.join(root, img) for img in imgs]
        # print('-------------------------------------------------------------')
        # print(imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        # print(img_path)
        # dog->1， cat->0
        # 以两个'/'之间的值为一个size，按着最后一个维度进行划分：/data/dagcat/cat.12345.jpg,在cat.12345.jpg处划分，一个cat.12345.jpg为一个size
        label = 1 if 'dog' in img_path.split('/')[-1] else 0
        # 例：划分出dogcat
        # print(img_path.split('/')[-2])
        pil_img = Image.open(img_path)
        # print(pil_img)
        array = np.asarray(pil_img)
        data = t.from_numpy(array)
        # print(data.shape)
        return data, label

    def __len__(self):
        return len(self.imgs)

dataset = DogCat('./data/dogcat/')
# print(dataset.imgs)
# 先用第一张图片'./data/dogcat/cat.12484.jpg'初始化img和label
img, label = dataset[0] # 相当于调用dataset.__getitem__(0)(这条语句没有用)

# for img, label in dataset:
#     print(img.size(), img.float().mean(), label)



# ImageFolder数据集
dataset = ImageFolder('data/dogcat_2/')
# 按着文件夹名命名label(从0开始)：cat:0 dog:1 pig:2
dataset.class_to_idx
# print(dataset.class_to_idx)
# 所有图片的路径和对应的label
dataset.imgs
# print(dataset.imgs)

# 此时还未进行任何的transform操作，所以返回的还是PIL Image对象
# 第一维是第几张图，第二维为1返回label
dataset[0][1]
# print(dataset[0][1])
# 第二维为0返回图片数据，返回的是Image对象
dataset[0][0]
# print(dataset[0][0])

# 使用transforms操作优化dataset
import os
from PIL import Image
import numpy as np
from torchvision import transforms as T

transform = T.Compose([
    T.Scale(224),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
    T.CenterCrop(224),  # 从图片中间切出224*224的图片
    T.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 1]
    T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至[-1, 1]，规定均值和标准差
])


class DogCat(data.Dataset):
    def __init__(self, root, transforms=None):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        # 调用transforms
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 0 if 'dog' in img_path.split('/')[-1] else 1
        data = Image.open(img_path)
        # 将数据进行transforms
        if self.transforms:
            data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


dataset = DogCat('./data/dogcat/', transforms=transform)
img, label = dataset[0]
# for img, label in dataset:
#     print(img.size(), label)



# 使用torchvision的ImageFolder自定义数据集
from torchvision.datasets import ImageFolder
dataset = ImageFolder('data/dogcat_2/')
dataset.class_to_idx
dataset.imgs
# 加上transform
normalize = T.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
transform  = T.Compose([
         T.RandomSizedCrop(224),
         T.RandomHorizontalFlip(),
         T.ToTensor(),
         normalize,
])
# 加上transforms后的ImageFolder
dataset = ImageFolder('data/dogcat_2/', transform=transform)
# 深度学习中图片数据一般保存成c * H * W，即通道数*图片高*图片宽
# print(dataset[0][0].size())
to_img = T.ToPILImage()
to_img(dataset[0][0]*0.2+0.4)# 0.2和0.4是标准差和均值的近似
# print(to_img(dataset[0][0]*0.2+0.4))

# 使用DataLoader加载数据
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate # 导入默认的样本拼接方式
dataloader = DataLoader(dataset,batch_size=3,shuffle=True,num_workers=0,drop_last=False)
# dataloader是个可迭代的对象
# 第一种迭代方法
dataiter = iter(dataloader)
imgs, labels = next(dataiter)
# print(imgs.size())
# 第二种迭代方法
# for batch_datas, batch_labels in dataloader:
#     train()


# 过滤掉数据集中错误的样本（如某张图片损坏）
class NewDogCat(DogCat):
    def __getitem__(self, index):
        # 将无法读取的样本抛出异常，返回None（data,label）对象
        try:
            # 调用父类的构造方法
            return super(NewDogCat,self).__getitem__(index)
        except:
            new_index = t.randint(0,len(self)-1)
            return self[new_index]

# 剔除掉损坏的样本后，dataloader返回的一个batch的样本数目会少于batch_size

    # 自定义collate_fn,将None对象过滤掉
# def my_collate_fn(batch):
#         '''
#         batch中每个元素形如（data，label）
#         '''
#
#         # 过滤为None的数据
#         batch = list(filter(lambda x:x[0] is not None, batch))
#         return default_collate(batch) # 用默认方式拼接过滤后的batch数据
dataset = NewDogCat('data/dogcat_wrong/', transforms=transform)
# print(dataset[-1])
dataloader = DataLoader(dataset, batch_size=2, num_workers=0)
dataiter = iter(dataloader)
imgs, labels = next(dataiter)
print(imgs.size(),labels.size())
# for batch_datas, batch_labels in dataloader:
#     print(batch_datas.size())

# WeightedRandomSampler：按样本权重采样
dataset = DogCat('data/dogcat/', transforms=transform)
# 狗的图片被取出的概率是猫的两倍
# 两类图片被取出的概率与weights的绝对大小无关，只与比值有关
weights = [2 if label == 1 else 1 for data, label in dataset]
print(weights)
# for data, label in dataset:
#     print(label)

from torch.utils.data.sampler import WeightedRandomSampler
sampler = WeightedRandomSampler(
            weights,
            num_samples=8,
            replacement=False
)
dataloader = DataLoader(
            dataset,
            batch_size=4,
            sampler=sampler
)

for datas, labels in dataloader:
    print(labels.tolist())

# torchvision视觉工具包，加载预训练的model，获取数据集
# from torchvision import models
# from torch import nn
# # 加载预训练好的模型，如果不存在就下载
# # 预训练好的模型保存在 ~/.torch/models/下
# resnet34 = models.resnet34(pretrained=True, num_classes = 1000)
# # 修改最后的全连接层为10分类问题（默认是ImageNet上的1000分类）
# resnet34.fc = nn.Linear(512,10)
# from torchvision import datasets
# # 指定数据集路径为data,如果数据集不存在则进行下载
# # 通过train = False获取测试集
# dataset = datasets.MNIST('data/',download=True,train=False,transform=transform)

from torchvision import transforms
import visdom
from matplotlib import pyplot as plt
# 随机产生一张有噪声的图片
to_pil = transforms.ToPILImage()
# Image.open('a.png').show()

vis = visdom.Visdom(env = u'test')
x = t.arange(1,30,0.01)
y = t.sin(x)
# vis.line(X=x, Y=y, win='sinx', opts={'title': 'y=sin(x)'})
# win表示窗格(pane)的问题，opts用来可视化配置，接收一个字典，主要用于设置pane的格式

# append追加数据
for ii in range(0, 10):
    x = t.Tensor([ii])
    y = x
    vis.line(X=x, Y=y, win='polynomial', update='append' if ii>0 else None)

# updateTrace 新增一条线
x = t.arange(0, 9, 0.01)
y = (x ** 2)/9
vis.line(X=x, Y=y, win='polynomial', update='append') # 通过指定update新增一条线，updateTrace方法已被弃用并删除
vis.image(t.randn(64,64).numpy())
vis.image(t.randn(3,256,256).numpy(), win ='random2')
vis.images(t.randn(8,3,32,32).numpy(), nrow = 4, win ='random3', opts={'title': 'random_imgs'})