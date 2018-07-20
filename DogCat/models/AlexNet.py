#coding:utf8
from torch import nn
from .BasicModule import BasicModule

class AlexNet(BasicModule):
    '''
    继承BasicModule,实现自定义的module
    code from torchvision/models/alexnet.py
    结构参考 <https://arxiv.org/abs/1404.5997>
    '''
    def __init__(self, num_classes=2):
        
        super(AlexNet, self).__init__()

        # 继承BasicModule的name属性
        self.model_name = 'alexnet'

        # 自定义alexnet的module

        # 对特征进行提取，卷积和ReLU,最后最大池化下采样
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # 对特征进行分类，全连接和ReLU
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    # 定义该module的前向传播，不需要定义backward
    def forward(self, x):
        x = self.features(x)
        # 特征层到分类层需要调整形状   由多通道展开为单通道
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
