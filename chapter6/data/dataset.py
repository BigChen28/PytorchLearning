#coding:utf8
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import  transforms as T



class DogCat(data.Dataset):
    
    def __init__(self,root,transforms=None,train=True,test=False):
        '''
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        '''
        self.test = test
        imgs = [os.path.join(root,img) for img in os.listdir(root)]

        # 加载图片名称，测试集名称以数字命名，训练集以名称.数字命名，如下：
        # test1: data/test1/8973.jpg
        # train: data/train/cat.10004.jpg 
        if self.test:
            # 划分测试集，先按 . 进行划分，再按 / 进行划分
            '''
            如：data/test1/8973.jpg
            第一次划分：data/test1/8973 和 jpg两部分,取[-2]中的 data/test1/8973 部分
            第二次划分：按照/划分， 结果为data test1 8973,取[-1]的 8973                        
            '''
            imgs = sorted(imgs,key=lambda x:int(x.split('.')[-2].split('/')[-1]))
        else:
            # 划分训练集，按 . 进行划分
            imgs = sorted(imgs,key=lambda x:int(x.split('.')[-2]))

        # imgs_num 代表图像的长度
        imgs_num = len(imgs)

        # 划分训练集和验证集。训练：验证 = 7:3
        if self.test:
            self.imgs = imgs
        # 训练集 前70%
        elif train:
            self.imgs = imgs[:int(0.7*imgs_num)]
        # 验证集 后30%
        else :
            self.imgs = imgs[int(0.7*imgs_num):]
            
        # 数据转换操作，测试验证和训练的数据转换有所区别
        if transforms is None:
            # 数据归一化
            normalize = T.Normalize(mean = [0.485, 0.456, 0.406], 
                                     std = [0.229, 0.224, 0.225])

            # 测试集和验证集的转换
            if self.test or not train: 
                self.transforms = T.Compose([
                    T.Scale(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                    ]) 
            # 训练集转换，数据增强
            else :
                self.transforms = T.Compose([
                    T.Scale(256),
                    T.RandomSizedCrop(224),  # 随机裁剪
                    T.RandomHorizontalFlip(),  # 随机水平翻转
                    T.ToTensor(),
                    normalize
                    ]) 
                
    # __getitem__方法，包含图片的读取，给图片设置标签，图片的转换等操作。
    def __getitem__(self,index):
        '''
        一次返回一张图片的数据
        '''
        img_path = self.imgs[index]
        # 如果是测试集，label = 数字
        if self.test: label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        # 给训练集和验证集设置标签 dog:1 cat:0
        else: label = 1 if 'dog' in img_path.split('/')[-1] else 0
        # 读取一张图片
        data = Image.open(img_path)
        # 对图片进行转换在·
        data = self.transforms(data)
        return data, label

    # 图片的长度
    def __len__(self):
        return len(self.imgs)