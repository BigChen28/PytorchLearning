from .AlexNet import AlexNet
from .ResNet34 import ResNet34
# from torchvision.models import InceptinV3
# from torchvision.models import alexnet as AlexNet

'''
    尽量使用nn.Sequential
    将经常使用的结构封装成子module
    将重复且有规律性的结构用函数生成，如load(),save()等
'''